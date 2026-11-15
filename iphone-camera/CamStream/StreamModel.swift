import AVFoundation
import Network
import Observation
import UIKit

/// État de l'app : relie la caméra, le serveur MJPEG et l'interface.
@MainActor
@Observable
final class StreamModel {
    // MARK: État

    private(set) var isRunning = false
    private(set) var serverState: MJPEGServer.State = .stopped
    private(set) var captureSource: CaptureSource?
    private(set) var viewerCount = 0
    private(set) var addresses: [String] = []
    private(set) var frameSize: CGSize = .zero
    private(set) var cameraAccessDenied = false
    /// Incrémenté après chaque (re)configuration de la caméra, pour rafraîchir l'aperçu.
    private(set) var configurationID = 0

    // MARK: Réglages (mémorisés entre les lancements)

    var position: AVCaptureDevice.Position {
        didSet {
            guard position != oldValue else { return }
            defaults.set(position == .front ? "front" : "back", forKey: Keys.position)
            pushStreamInfo()
            restartCaptureIfRunning()
        }
    }

    /// Lens asked for on the back camera (the front one is always 1×).
    var lens: CameraLens {
        didSet {
            guard lens != oldValue else { return }
            defaults.set(lens.rawValue, forKey: Keys.lens)
            restartCaptureIfRunning()
        }
    }

    /// Capture the LiDAR depth with the video (back camera of a Pro iPhone, 1× lens).
    var lidar: Bool {
        didSet {
            guard lidar != oldValue else { return }
            defaults.set(lidar, forKey: Keys.lidar)
            restartCaptureIfRunning()
        }
    }

    var resolution: StreamResolution {
        didSet {
            guard resolution != oldValue else { return }
            defaults.set(resolution.rawValue, forKey: Keys.resolution)
            restartCaptureIfRunning()
        }
    }

    var fps: Int {
        didSet {
            defaults.set(fps, forKey: Keys.fps)
            pushEncoderSettings()
            pushStreamInfo()
        }
    }

    var quality: JPEGQuality {
        didSet {
            defaults.set(quality.rawValue, forKey: Keys.quality)
            pushEncoderSettings()
        }
    }

    static let fpsOptions = [15, 24, 30]

    /// The back camera has an ultra wide lens (not on the SE nor before the iPhone 11).
    let ultraWideAvailable = CameraManager.hasUltraWideCamera
    /// The iPhone has a LiDAR scanner (Pro models from the iPhone 12 Pro).
    let lidarAvailable = CameraManager.hasLiDAR

    let camera = CameraManager()
    @ObservationIgnored private let server = MJPEGServer()
    @ObservationIgnored private let pathMonitor = NWPathMonitor()
    @ObservationIgnored private let defaults = UserDefaults.standard

    private enum Keys {
        static let position = "position"
        static let lens = "lens"
        static let lidar = "lidar"
        static let resolution = "resolution"
        static let fps = "fps"
        static let quality = "quality"
    }

    init() {
        position = defaults.string(forKey: Keys.position) == "front" ? .front : .back
        lens = CameraLens(rawValue: defaults.string(forKey: Keys.lens) ?? "") ?? .wide
        lidar = defaults.bool(forKey: Keys.lidar)
        resolution = StreamResolution(rawValue: defaults.string(forKey: Keys.resolution) ?? "") ?? .hd720
        let savedFPS = defaults.integer(forKey: Keys.fps)
        fps = Self.fpsOptions.contains(savedFPS) ? savedFPS : 30
        quality = JPEGQuality(rawValue: defaults.string(forKey: Keys.quality) ?? "") ?? .medium

        let server = self.server
        let camera = self.camera
        camera.hasViewers = { server.hasViewers }
        camera.onFrame = { jpeg in server.broadcast(jpeg) }
        server.depthProvider = { camera.latestDepth() }
        server.depthPreview = { camera.depthPreviewJPEG() }
        camera.onFrameSizeChange = { [weak self] size in
            guard let self else { return }
            Self.onMain {
                self.frameSize = size
                self.pushStreamInfo()
            }
        }

        server.onStateChange = { [weak self] state in
            guard let self else { return }
            Self.onMain { self.serverState = state }
        }
        server.onClientCountChange = { [weak self] count in
            guard let self else { return }
            Self.onMain { self.viewerCount = count }
        }

        pathMonitor.pathUpdateHandler = { [weak self] _ in
            guard let self else { return }
            Self.onMain { self.refreshAddresses() }
        }
        pathMonitor.start(queue: DispatchQueue(label: "camstream.path"))

        pushEncoderSettings()
        pushStreamInfo()
        refreshAddresses()
    }

    // MARK: Valeurs dérivées

    var port: UInt16 {
        if case .running(let port) = serverState { return port }
        return 8080
    }

    var isLive: Bool {
        guard isRunning, case .running = serverState else { return false }
        switch captureSource {
        case .camera, .testPattern: return true
        default: return false
        }
    }

    /// Adresses de la page web, une par interface réseau (Wi-Fi en premier).
    var pageURLs: [String] {
        addresses.map { "http://\($0):\(port)" }
    }

    var primaryURL: String? { pageURLs.first }

    /// Lens of the camera being streamed; nil without a camera (test pattern, failure).
    var activeLens: CameraLens? {
        if case .camera(_, let lens, _) = captureSource { return lens }
        return nil
    }

    /// True while the LiDAR depth is captured with the video.
    var depthActive: Bool {
        if case .camera(_, _, let depth) = captureSource { return depth }
        return false
    }

    /// The lens button and picker only make sense on the back camera of an iPhone that has
    /// both, and not with the LiDAR on (it only works with the 1× lens).
    var canChooseLens: Bool {
        ultraWideAvailable && position == .back && !lidar && captureSource != .testPattern
    }

    /// The LiDAR switch: back camera of an iPhone that has one.
    var canUseLiDAR: Bool {
        lidarAvailable && position == .back && captureSource != .testPattern
    }

    var statusText: String {
        if !isRunning { return "Arrêté" }
        if case .failed(let message) = captureSource { return message }
        if case .failed(let message) = serverState { return message }
        if isLive {
            if captureSource == .testPattern { return "En direct · mire" }
            if depthActive { return "En direct · LiDAR" }
            return activeLens == .ultraWide ? "En direct · grand angle" : "En direct"
        }
        return "Démarrage…"
    }

    // MARK: Actions

    func start() async {
        guard !isRunning else { return }
        isRunning = true
        cameraAccessDenied = false

        guard await Self.requestCameraAccess() else {
            cameraAccessDenied = true
            isRunning = false
            return
        }
        guard isRunning else { return } // arrêté pendant la demande d'autorisation

        UIApplication.shared.isIdleTimerDisabled = true // l'écran ne se verrouille pas pendant la diffusion
        refreshAddresses()
        server.start(port: 8080)
        startCapture()
    }

    func stop() {
        guard isRunning else { return }
        isRunning = false
        camera.stop()
        server.stop()
        captureSource = nil
        viewerCount = 0
        UIApplication.shared.isIdleTimerDisabled = false
    }

    func toggle() {
        if isRunning {
            stop()
        } else {
            Task { await start() }
        }
    }

    func switchCamera() {
        position = position == .back ? .front : .back
    }

    func switchLens() {
        lens = lens == .wide ? .ultraWide : .wide
    }

    /// Retour au premier plan : iOS a pu fermer le socket du serveur pendant la mise en veille.
    func sceneBecameActive() {
        refreshAddresses()
        guard isRunning else { return }
        server.ensureRunning()
    }

    // MARK: Interne

    private func startCapture() {
        camera.start(position: position, lens: lens, lidar: lidar,
                     resolution: resolution) { [weak self] source in
            guard let self else { return }
            Self.onMain {
                guard self.isRunning else { return }
                self.captureSource = source
                self.configurationID += 1
                self.pushStreamInfo() // the lens actually in use, for /status
            }
        }
    }

    private func restartCaptureIfRunning() {
        guard isRunning else { return }
        startCapture()
    }

    private func refreshAddresses() {
        addresses = NetworkInfo.localIPv4Addresses()
    }

    private func pushEncoderSettings() {
        camera.updateEncoder(fps: fps, quality: quality)
    }

    private func pushStreamInfo() {
        server.updateInfo(StreamInfo(
            width: Int(frameSize.width),
            height: Int(frameSize.height),
            fps: fps,
            camera: position == .front ? "front" : "back",
            lens: activeLens?.rawValue ?? "none",
            depth: depthActive ? "lidar" : (lidarAvailable ? "off" : "unavailable")
        ))
    }

    private static func requestCameraAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }

    /// Exécute sur le thread principal, dans l'ordre d'arrivée.
    private nonisolated static func onMain(_ body: @escaping @MainActor @Sendable () -> Void) {
        DispatchQueue.main.async {
            MainActor.assumeIsolated { body() }
        }
    }
}
