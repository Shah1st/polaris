import ARKit
import AVFoundation
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
    /// The LiDAR depth is captured with the pose (`start(depth: true)` on a Pro iPhone).
    private var wantsDepth = false
    private var lastDepthTime: CFTimeInterval = 0
    /// Depth maps are converted at most this often (s).
    private let depthPeriod: CFTimeInterval = 0.2
    /// The newest depth map and the camera's pose for it.
    private let depth = OSAllocatedUnfairLock<DepthFrame?>(initialState: nil)

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = queue
    }

    func updateEncoder(fps: Int, quality: JPEGQuality) {
        settings.withLock { $0 = EncoderSettings(maxFPS: fps, quality: quality.compression) }
    }

    /// The lenses ARKit can track with on this iPhone (its video formats' cameras).
    static var trackableLenses: Set<CameraLens> {
        Set(ARWorldTrackingConfiguration.supportedVideoFormats.compactMap { lens(of: $0) })
    }

    private static func lens(of format: ARConfiguration.VideoFormat) -> CameraLens? {
        switch format.captureDeviceType {
        case .builtInUltraWideCamera: .ultraWide
        case .builtInWideAngleCamera: .wide
        default: nil
        }
    }

    /// ARKit gives the LiDAR depth with the pose on this iPhone (Pro models).
    static var hasSceneDepth: Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    /// The newest LiDAR depth map with its pose, or nil (no LiDAR, depth off, not running).
    func latestDepth() -> DepthFrame? {
        depth.withLock { $0 }
    }

    /// Starts (or restarts) the tracking: the world frame is reset where the phone is now.
    /// `lens`: the lens asked for; ARKit's formats decide whether it is possible. `depth`: the
    /// LiDAR depth too (it comes with the 1× lens only). `completion` gets the lens actually
    /// used and whether the depth is captured.
    func start(resolution: StreamResolution, lens: CameraLens, depth wantDepth: Bool,
               completion: @escaping @Sendable (CameraLens, Bool) -> Void) {
        queue.async { [self] in
            targetWidth = resolution.testPatternSize.width
            planes.removeAll()
            lastFrameSize = .zero
            sessionID = Int.random(in: 1...999_999)
            let configuration = ARWorldTrackingConfiguration()
            configuration.worldAlignment = .gravity // y up, whatever the phone's attitude
            configuration.planeDetection = [.horizontal] // to find the floor
            let withDepth = wantDepth && Self.hasSceneDepth
            let lens: CameraLens = withDepth ? .wide : lens // the LiDAR's depth is the 1× lens's
            // The format of the wanted lens nearest to the chosen resolution, at least as wide.
            let formats = ARWorldTrackingConfiguration.supportedVideoFormats
                .filter { Self.lens(of: $0) == lens }
            let wide = formats.filter { $0.imageResolution.width >= targetWidth }
            if let format = (wide.isEmpty ? formats : wide)
                .min(by: { abs($0.imageResolution.width - targetWidth) < abs($1.imageResolution.width - targetWidth) }) {
                configuration.videoFormat = format
            }
            if withDepth {
                // smoothed over a few frames: steadier heights for a map
                if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
                    configuration.frameSemantics.insert(.smoothedSceneDepth)
                } else {
                    configuration.frameSemantics.insert(.sceneDepth)
                }
            }
            wantsDepth = withDepth
            depth.withLock { $0 = nil }
            let used = Self.lens(of: configuration.videoFormat) ?? .wide
            session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
            running = true
            completion(used, withDepth)
        }
    }

    /// `completion` runs once ARKit has let go of the camera (at once when it was not running).
    func stop(completion: (@Sendable () -> Void)? = nil) {
        queue.async { [self] in
            if running {
                running = false
                session.pause()
            }
            depth.withLock { $0 = nil }
            completion?()
        }
    }

    // MARK: - ARSessionDelegate (queue)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard running else { return }
        if wantsDepth { storeDepth(frame) } // even when no picture is encoded
        guard shouldEncodeFrame() else { return }

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

    /// Keeps the newest LiDAR depth (Float32 metres along the optical axis, the points of low
    /// confidence set to 0) with the pose of its frame, at most every `depthPeriod`.
    private func storeDepth(_ frame: ARFrame) {
        let now = CACurrentMediaTime()
        guard now - lastDepthTime >= depthPeriod,
              let scene = frame.smoothedSceneDepth ?? frame.sceneDepth else { return }
        lastDepthTime = now

        let map = scene.depthMap
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return }
        let width = CVPixelBufferGetWidth(map), height = CVPixelBufferGetHeight(map)
        let rowBytes = CVPixelBufferGetBytesPerRow(map)

        let confidence = scene.confidenceMap
        if let confidence { CVPixelBufferLockBaseAddress(confidence, .readOnly) }
        defer { if let confidence { CVPixelBufferUnlockBaseAddress(confidence, .readOnly) } }
        let confidenceBase = confidence.flatMap { CVPixelBufferGetBaseAddress($0) }
        let confidenceRow = confidence.map { CVPixelBufferGetBytesPerRow($0) } ?? 0
        let medium = UInt8(ARConfidenceLevel.medium.rawValue)

        var metres = Data(count: width * height * MemoryLayout<Float32>.size)
        metres.withUnsafeMutableBytes { (out: UnsafeMutableRawBufferPointer) in
            let dst = out.bindMemory(to: Float32.self)
            for row in 0..<height {
                let src = (base + row * rowBytes).assumingMemoryBound(to: Float32.self)
                let conf = confidenceBase.map {
                    ($0 + row * confidenceRow).assumingMemoryBound(to: UInt8.self)
                }
                for col in 0..<width {
                    var d = src[col]
                    if let conf, conf[col] < medium { d = 0 }
                    dst[row * width + col] = d
                }
            }
        }

        let resolution = frame.camera.imageResolution
        let k = frame.camera.intrinsics
        let sx = Double(width) / Double(resolution.width)
        let sy = Double(height) / Double(resolution.height)
        let intrinsics = [Double(k.columns.0.x) * sx, Double(k.columns.1.y) * sy,
                          Double(k.columns.2.x) * sx, Double(k.columns.2.y) * sy]
        let scale = min(1, targetWidth / max(resolution.width, 1))
        let size = CGSize(width: (resolution.width * scale).rounded(),
                          height: (resolution.height * scale).rounded())
        let pose = poseJSON(frame: frame, size: size, scale: Float(scale))
        let stored = DepthFrame(width: width, height: height, metres: metres,
                                intrinsics: intrinsics, videoRotation: 0, time: now, pose: pose)
        depth.withLock { $0 = stored }
    }

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
