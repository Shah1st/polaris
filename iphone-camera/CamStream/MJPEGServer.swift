import Foundation
import Network
import os

/// Informations publiées sur `/status`.
struct StreamInfo: Codable, Sendable, Equatable {
    var width = 0
    var height = 0
    var fps = 30
    var camera = "back"
}

/// Petit serveur HTTP qui diffuse les images JPEG de la caméra en MJPEG
/// (`multipart/x-mixed-replace`), un format lisible par tous les navigateurs,
/// VLC, OBS, ffmpeg, OpenCV…
///
/// Tout l'état interne est manipulé uniquement sur `queue`.
final class MJPEGServer: @unchecked Sendable {
    enum State: Equatable, Sendable {
        case stopped
        case starting
        case running(port: UInt16)
        case failed(String)
    }

    static let boundary = "camstreamframe"

    let serviceName: String
    var onStateChange: (@Sendable (State) -> Void)?
    var onClientCountChange: (@Sendable (Int) -> Void)?

    private let queue = DispatchQueue(label: "camstream.server", qos: .userInitiated)
    private var listener: NWListener?
    private var shouldRun = false
    private var preferredPort: UInt16 = 8080
    private var portAttempts = 0
    private var streamClients: [ObjectIdentifier: StreamClient] = [:]
    private var latestFrame: Data?
    private var info = StreamInfo()
    private let viewerCount = OSAllocatedUnfairLock(initialState: 0)

    init(serviceName: String = "CamStream") {
        self.serviceName = serviceName
    }

    /// Lisible depuis n'importe quel thread (utilisé par l'encodeur pour adapter sa cadence).
    var hasViewers: Bool { viewerCount.withLock { $0 > 0 } }

    // MARK: - Cycle de vie

    func start(port: UInt16 = 8080) {
        queue.async { [self] in
            shouldRun = true
            guard listener == nil else { return }
            preferredPort = port
            portAttempts = 0
            startListener(on: port)
        }
    }

    /// À appeler au retour au premier plan : iOS peut avoir fermé le socket d'écoute.
    func ensureRunning() {
        queue.async { [self] in
            guard shouldRun else { return }
            if let listener {
                switch listener.state {
                case .failed, .cancelled: break
                default: return
                }
                listener.cancel()
                self.listener = nil
            }
            startListener(on: preferredPort)
        }
    }

    func stop() {
        queue.async { [self] in
            shouldRun = false
            listener?.cancel()
            listener = nil
            for client in streamClients.values { client.connection.cancel() }
            streamClients.removeAll()
            latestFrame = nil
            updateViewerCount()
            report(.stopped)
        }
    }

    /// Envoie une nouvelle image à tous les spectateurs.
    func broadcast(_ jpeg: Data) {
        queue.async { [self] in
            latestFrame = jpeg
            for client in streamClients.values { client.enqueue(jpeg) }
        }
    }

    func updateInfo(_ newInfo: StreamInfo) {
        queue.async { [self] in info = newInfo }
    }

    // MARK: - Écoute

    private func startListener(on port: UInt16) {
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true

        guard let nwPort = NWEndpoint.Port(rawValue: port),
              let listener = try? NWListener(using: parameters, on: nwPort)
        else {
            report(.failed("Impossible d'ouvrir le port \(port)."))
            return
        }

        // Annonce Bonjour : le service apparaît comme « CamStream » (_http._tcp) sur le réseau.
        listener.service = NWListener.Service(
            name: serviceName,
            type: "_http._tcp",
            domain: nil,
            txtRecord: NWTXTRecord(["path": "/"])
        )
        listener.stateUpdateHandler = { [weak self] state in
            self?.listenerStateChanged(state, port: port)
        }
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }

        self.listener = listener
        report(.starting)
        listener.start(queue: queue)
    }

    private func listenerStateChanged(_ state: NWListener.State, port: UInt16) {
        switch state {
        case .ready:
            report(.running(port: listener?.port?.rawValue ?? port))
        case .failed(let error):
            listener?.cancel()
            listener = nil
            guard shouldRun else { return }
            if case .posix(let code) = error, code == .EADDRINUSE, portAttempts < 10 {
                // Port occupé : on essaie le suivant.
                portAttempts += 1
                preferredPort = port + 1
                startListener(on: preferredPort)
            } else {
                report(.failed(error.localizedDescription))
            }
        case .waiting(let error):
            report(.failed("En attente du réseau : \(error.localizedDescription)"))
        default:
            break
        }
    }

    private func report(_ state: State) {
        onStateChange?(state)
    }

    // MARK: - Connexions

    private func accept(_ connection: NWConnection) {
        let id = ObjectIdentifier(connection)
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled:
                self?.connectionClosed(id)
            default:
                break
            }
        }
        connection.start(queue: queue)
        receiveRequest(on: connection, buffer: Data())
    }

    private func connectionClosed(_ id: ObjectIdentifier) {
        if streamClients.removeValue(forKey: id) != nil {
            updateViewerCount()
        }
    }

    private func receiveRequest(on connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 16 * 1024) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }

            if let headerEnd = buffer.range(of: Data("\r\n\r\n".utf8)) {
                let head = String(decoding: buffer[..<headerEnd.lowerBound], as: UTF8.self)
                self.handle(requestHead: head, on: connection)
            } else if isComplete || error != nil || buffer.count > 64 * 1024 {
                connection.cancel()
            } else {
                self.receiveRequest(on: connection, buffer: buffer)
            }
        }
    }

    private func handle(requestHead: String, on connection: NWConnection) {
        let requestLine = requestHead.components(separatedBy: "\r\n").first ?? ""
        let parts = requestLine.split(separator: " ")
        guard parts.count >= 2 else {
            sendResponse(on: connection, status: "400 Bad Request", contentType: "text/plain; charset=utf-8", body: Data("Requête invalide".utf8))
            return
        }

        let method = String(parts[0])
        var path = String(parts[1])
        if let query = path.firstIndex(of: "?") { path = String(path[..<query]) }

        guard method == "GET" || method == "HEAD" else {
            sendResponse(on: connection, status: "405 Method Not Allowed", contentType: "text/plain; charset=utf-8", body: Data())
            return
        }
        let headOnly = method == "HEAD"

        switch path {
        case "/", "/index.html":
            sendResponse(on: connection, status: "200 OK", contentType: "text/html; charset=utf-8",
                         body: Data(WebPage.player.utf8), headOnly: headOnly)

        case "/view":
            sendResponse(on: connection, status: "200 OK", contentType: "text/html; charset=utf-8",
                         body: Data(WebPage.bare.utf8), headOnly: headOnly)

        case "/stream", "/stream.mjpg", "/video", "/mjpeg":
            if headOnly {
                sendResponse(on: connection, status: "200 OK",
                             contentType: "multipart/x-mixed-replace; boundary=\(Self.boundary)", body: Data(), headOnly: true)
            } else {
                startStream(on: connection)
            }

        case "/snapshot", "/snapshot.jpg", "/photo.jpg", "/shot.jpg":
            if let latestFrame {
                sendResponse(on: connection, status: "200 OK", contentType: "image/jpeg", body: latestFrame, headOnly: headOnly)
            } else {
                sendResponse(on: connection, status: "503 Service Unavailable", contentType: "text/plain; charset=utf-8",
                             body: Data("Aucune image disponible pour le moment.".utf8), headOnly: headOnly)
            }

        case "/status", "/status.json":
            struct Status: Encodable {
                let viewers: Int
                let width: Int
                let height: Int
                let fps: Int
                let camera: String
            }
            let status = Status(viewers: streamClients.count, width: info.width, height: info.height,
                                fps: info.fps, camera: info.camera)
            let body = (try? JSONEncoder().encode(status)) ?? Data("{}".utf8)
            sendResponse(on: connection, status: "200 OK", contentType: "application/json", body: body, headOnly: headOnly)

        case "/favicon.ico":
            sendResponse(on: connection, status: "204 No Content", contentType: "image/x-icon", body: Data())

        default:
            sendResponse(on: connection, status: "404 Not Found", contentType: "text/plain; charset=utf-8",
                         body: Data("Introuvable. Essayez / , /stream ou /snapshot.jpg".utf8), headOnly: headOnly)
        }
    }

    private func sendResponse(on connection: NWConnection, status: String, contentType: String,
                              body: Data, headOnly: Bool = false) {
        var head = "HTTP/1.1 \(status)\r\n"
        head += "Content-Type: \(contentType)\r\n"
        head += "Content-Length: \(body.count)\r\n"
        head += "Cache-Control: no-cache, no-store, must-revalidate\r\n"
        head += "Access-Control-Allow-Origin: *\r\n"
        head += "Connection: close\r\n\r\n"

        var payload = Data(head.utf8)
        if !headOnly { payload.append(body) }
        connection.send(content: payload, isComplete: true, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }

    private func startStream(on connection: NWConnection) {
        var head = "HTTP/1.1 200 OK\r\n"
        head += "Content-Type: multipart/x-mixed-replace; boundary=\(Self.boundary)\r\n"
        head += "Cache-Control: no-cache, no-store, must-revalidate\r\n"
        head += "Pragma: no-cache\r\n"
        head += "Access-Control-Allow-Origin: *\r\n"
        head += "Connection: close\r\n\r\n"

        let client = StreamClient(connection: connection)
        streamClients[ObjectIdentifier(connection)] = client
        updateViewerCount()

        client.sendHeader(Data(head.utf8))
        if let latestFrame { client.enqueue(latestFrame) }
        watchForDisconnect(connection)
    }

    /// Le client n'envoie plus rien après sa requête : une lecture qui se termine = déconnexion.
    private func watchForDisconnect(_ connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1024) { [weak self] _, _, isComplete, error in
            if isComplete || error != nil {
                connection.cancel()
            } else {
                self?.watchForDisconnect(connection)
            }
        }
    }

    private func updateViewerCount() {
        let count = streamClients.count
        viewerCount.withLock { $0 = count }
        onClientCountChange?(count)
    }
}

/// Un spectateur du flux MJPEG. Si le réseau est lent, les images intermédiaires
/// sont ignorées : on n'envoie que la plus récente (pas d'accumulation de latence).
private final class StreamClient: @unchecked Sendable {
    let connection: NWConnection
    private var isSending = false
    private var pendingFrame: Data?

    init(connection: NWConnection) {
        self.connection = connection
    }

    func sendHeader(_ header: Data) {
        isSending = true
        connection.send(content: header, completion: .contentProcessed { [weak self] error in
            self?.didSend(error: error)
        })
    }

    func enqueue(_ frame: Data) {
        if isSending {
            pendingFrame = frame
        } else {
            send(frame)
        }
    }

    private func send(_ frame: Data) {
        isSending = true
        var chunk = Data()
        chunk.reserveCapacity(frame.count + 128)
        chunk.append(Data("--\(MJPEGServer.boundary)\r\nContent-Type: image/jpeg\r\nContent-Length: \(frame.count)\r\n\r\n".utf8))
        chunk.append(frame)
        chunk.append(Data("\r\n".utf8))
        connection.send(content: chunk, completion: .contentProcessed { [weak self] error in
            self?.didSend(error: error)
        })
    }

    private func didSend(error: NWError?) {
        isSending = false
        if error != nil {
            connection.cancel()
            return
        }
        if let next = pendingFrame {
            pendingFrame = nil
            send(next)
        }
    }
}
