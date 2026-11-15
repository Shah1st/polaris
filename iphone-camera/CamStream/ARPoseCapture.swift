import ARKit
import CoreImage
import ImageIO
import QuartzCore
import os

/// Captures the back camera through ARKit, which also tracks where the phone is in the room.
/// Each picture leaves with the pose of the camera when it was taken (one line of JSON, sent
/// as the `X-Pose` header, see the README): a program can then place what it sees on the
/// floor in metres, whatever the phone's movements.
///
/// The picture is the sensor's own (landscape) and is NOT rotated with the phone, so that the
/// intrinsics sent with it stay true. Everything but `start`, `stop` and `updateEncoder` runs
/// on `queue`.
final class ARPoseCapture: NSObject, ARSessionDelegate, @unchecked Sendable {
    /// ARKit world tracking exists on this iPhone (every iPhone since the 6s).
    static var isSupported: Bool { ARWorldTrackingConfiguration.isSupported }

    var hasViewers: @Sendable () -> Bool = { false }
    /// A JPEG and the pose of the camera for that very picture.
    var onFrame: (@Sendable (Data, String) -> Void)?
    var onFrameSizeChange: (@Sendable (CGSize) -> Void)?
    var onFailure: (@Sendable (String) -> Void)?

    private struct EncoderSettings: Sendable {
        var maxFPS = 30
        var quality: CGFloat = 0.6
    }

    private struct Plane {
        var y: Float
        var area: Float
    }

    private let queue = DispatchQueue(label: "camstream.arkit", qos: .userInitiated)
    private let session = ARSession()
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private let settings = OSAllocatedUnfairLock(initialState: EncoderSettings())

    // queue only
    private var running = false
    private var targetWidth: CGFloat = 1280
    /// Changes at every start: the world frame of ARKit starts again where the phone is then.
    private var sessionID = 0
    private var nextFrameTime: CFTimeInterval = 0
    private var lastFrameSize: CGSize = .zero
    /// Horizontal planes ARKit has found so far (the floor, tables…).
    private var planes: [UUID: Plane] = [:]

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = queue
    }

    func updateEncoder(fps: Int, quality: JPEGQuality) {
        settings.withLock { $0 = EncoderSettings(maxFPS: fps, quality: quality.compression) }
    }

    /// Starts (or restarts) the tracking: the world frame is reset where the phone is now.
    func start(resolution: StreamResolution) {
        queue.async { [self] in
            targetWidth = resolution.testPatternSize.width
            planes.removeAll()
            lastFrameSize = .zero
            sessionID = Int.random(in: 1...999_999)
            let configuration = ARWorldTrackingConfiguration()
            configuration.worldAlignment = .gravity // y up, whatever the phone's attitude
            configuration.planeDetection = [.horizontal] // to find the floor
            session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
            running = true
        }
    }

    /// `completion` runs once ARKit has let go of the camera (at once when it was not running).
    func stop(completion: (@Sendable () -> Void)? = nil) {
        queue.async { [self] in
            if running {
                running = false
                session.pause()
            }
            completion?()
        }
    }

    // MARK: - ARSessionDelegate (queue)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard running, shouldEncodeFrame() else { return }

        let resolution = frame.camera.imageResolution
        let scale = min(1, targetWidth / max(resolution.width, 1))
        var image = CIImage(cvPixelBuffer: frame.capturedImage)
        if scale < 1 {
            image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }

        let quality = settings.withLock { $0.quality }
        let key = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        guard let jpeg = ciContext.jpegRepresentation(of: image, colorSpace: colorSpace, options: [key: quality]) else {
            return
        }

        let size = CGSize(width: image.extent.width.rounded(), height: image.extent.height.rounded())
        if size != lastFrameSize {
            lastFrameSize = size
            onFrameSizeChange?(size)
        }
        onFrame?(jpeg, poseJSON(frame: frame, size: size, scale: Float(scale)))
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        store(anchors)
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        store(anchors)
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for anchor in anchors {
            planes[anchor.identifier] = nil
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        running = false
        onFailure?("ARKit : \(error.localizedDescription)")
    }

    // MARK: - Interne (queue)

    private func store(_ anchors: [ARAnchor]) {
        for anchor in anchors {
            guard let plane = anchor as? ARPlaneAnchor, plane.alignment == .horizontal else { continue }
            let centre = plane.transform * SIMD4<Float>(plane.center.x, plane.center.y, plane.center.z, 1)
            planes[plane.identifier] = Plane(y: centre.y,
                                             area: plane.planeExtent.width * plane.planeExtent.height)
        }
    }

    /// The floor: the lowest horizontal plane under the camera, among those of at least
    /// 0.25 m² when there are some (a small low plane is often a mistake).
    private func floorPlane(below cameraY: Float) -> Plane? {
        let below = planes.values.filter { $0.y < cameraY - 0.3 }
        let large = below.filter { $0.area >= 0.25 }
        return (large.isEmpty ? below : large).min { $0.y < $1.y }
    }

    /// One line of JSON, see the README: time, session, size and intrinsics of the JPEG,
    /// camera-to-world transform (row after row), tracking state, height of the floor.
    private func poseJSON(frame: ARFrame, size: CGSize, scale: Float) -> String {
        let camera = frame.camera
        let k = camera.intrinsics
        let m = camera.transform

        var transform: [String] = []
        for row in 0..<4 {
            for column in [m.columns.0, m.columns.1, m.columns.2, m.columns.3] {
                transform.append(String(format: "%.6f", column[row]))
            }
        }
        let intrinsics = [k.columns.0.x, k.columns.1.y, k.columns.2.x, k.columns.2.y]
            .map { String(format: "%.3f", $0 * scale) }

        let track: String
        switch camera.trackingState {
        case .normal:
            track = "normal"
        case .notAvailable:
            track = "none"
        case .limited(let reason):
            switch reason {
            case .initializing: track = "limited:initializing"
            case .excessiveMotion: track = "limited:motion"
            case .insufficientFeatures: track = "limited:features"
            case .relocalizing: track = "limited:relocalizing"
            @unknown default: track = "limited"
            }
        }

        var floorY = "null"
        var floorArea = "null"
        if let floor = floorPlane(below: m.columns.3.y) {
            floorY = String(format: "%.4f", floor.y)
            floorArea = String(format: "%.2f", floor.area)
        }

        return "{\"t\":\(String(format: "%.4f", frame.timestamp)),\"sid\":\(sessionID),"
            + "\"w\":\(Int(size.width)),\"h\":\(Int(size.height)),"
            + "\"k\":[\(intrinsics.joined(separator: ","))],"
            + "\"T\":[\(transform.joined(separator: ","))],"
            + "\"track\":\"\(track)\",\"floor_y\":\(floorY),\"floor_area\":\(floorArea)}"
    }

    /// Limits the encoding rate to the chosen frames per second (1 a second without viewers).
    private func shouldEncodeFrame() -> Bool {
        let now = CACurrentMediaTime()
        let fps = hasViewers() ? Double(settings.withLock { $0.maxFPS }) : 1
        let interval = 1 / max(fps, 1)
        guard now + 0.004 >= nextFrameTime else { return false }
        nextFrameTime = max(nextFrameTime + interval, now)
        return true
    }
}
