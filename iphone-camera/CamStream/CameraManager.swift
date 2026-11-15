import AVFoundation
import CoreImage
import ImageIO
import QuartzCore
import os
import simd

enum StreamResolution: String, CaseIterable, Identifiable, Sendable {
    case sd480 = "480p"
    case hd720 = "720p"
    case hd1080 = "1080p"

    var id: String { rawValue }

    var preset: AVCaptureSession.Preset {
        switch self {
        case .sd480: .vga640x480
        case .hd720: .hd1280x720
        case .hd1080: .hd1920x1080
        }
    }

    var testPatternSize: CGSize {
        switch self {
        case .sd480: CGSize(width: 640, height: 480)
        case .hd720: CGSize(width: 1280, height: 720)
        case .hd1080: CGSize(width: 1920, height: 1080)
        }
    }
}

enum JPEGQuality: String, CaseIterable, Identifiable, Sendable {
    case low = "Basse"
    case medium = "Moyenne"
    case high = "Haute"

    var id: String { rawValue }

    var compression: CGFloat {
        switch self {
        case .low: 0.4
        case .medium: 0.6
        case .high: 0.8
        }
    }
}

/// Lens of the back camera. Each one is a physical camera, never the multi-lens virtual camera,
/// which switches lens by itself (focus distance, light) and would change the picture's scale
/// in the middle of a run.
enum CameraLens: String, CaseIterable, Identifiable, Sendable {
    /// Main lens, 1× (Apple calls it the "wide angle" camera).
    case wide
    /// Ultra wide lens, 0.5×: about twice the field of view. iOS corrects its distortion by
    /// default (`isGeometricDistortionCorrectionEnabled`).
    case ultraWide = "ultrawide"

    var id: String { rawValue }

    /// Short label, as in Apple's Camera app.
    var label: String {
        switch self {
        case .wide: "1×"
        case .ultraWide: "0,5×"
        }
    }

    var title: String {
        switch self {
        case .wide: "Standard 1×"
        case .ultraWide: "Grand angle 0,5×"
        }
    }

    var deviceType: AVCaptureDevice.DeviceType {
        switch self {
        case .wide: .builtInWideAngleCamera
        case .ultraWide: .builtInUltraWideCamera
        }
    }
}

/// Video stabilization asked for. iOS steadies a handheld picture by cropping and shifting it:
/// the field of view narrows (the scale of a map seen from above changes) and the cinematic
/// modes delay the stream by a fraction of a second.
enum VideoStabilization: String, CaseIterable, Identifiable, Sendable {
    case off
    case standard
    case cinematic
    /// The strongest mode this iPhone accepts with this lens and resolution.
    case maximum

    var id: String { rawValue }

    var title: String {
        switch self {
        case .off: "Aucune"
        case .standard: "Standard"
        case .cinematic: "Cinéma"
        case .maximum: "Maximum"
        }
    }

    /// One iOS mode and its name for `/status`.
    struct Choice {
        let mode: AVCaptureVideoStabilizationMode
        let name: String
    }

    /// iOS modes to try, strongest first: the first one the camera's format supports is used.
    var choices: [Choice] {
        let standard = Choice(mode: .standard, name: "standard")
        let cinematic = Choice(mode: .cinematic, name: "cinematic")
        let extended = Choice(mode: .cinematicExtended, name: "cinematic-extended")
        switch self {
        case .off:
            return []
        case .standard:
            return [standard]
        case .cinematic:
            return [cinematic, standard]
        case .maximum:
            if #available(iOS 18.0, *) {
                let enhanced = Choice(mode: .cinematicExtendedEnhanced,
                                      name: "cinematic-extended-enhanced")
                return [enhanced, extended, cinematic, standard]
            }
            return [extended, cinematic, standard]
        }
    }
}

/// The newest LiDAR depth map, ready to be served (`/depth`).
struct DepthFrame: Sendable {
    let width: Int
    let height: Int
    /// Distance along the optical axis in metres, Float32 little-endian, row after row from the
    /// top left of the sensor's own (landscape) orientation; 0 or NaN where there is no measure.
    let metres: Data
    /// fx, fy, cx, cy in depth pixels; nil when iOS gives no calibration with the depth.
    let intrinsics: [Double]?
    /// Rotation applied to the video at that time (`videoRotationAngle`, degrees). The depth is
    /// NOT rotated: turn it the same way to lay it over the video.
    let videoRotation: Double
    /// `CACurrentMediaTime()` when the depth arrived.
    let time: CFTimeInterval
    /// ARKit mode: the camera's pose for that depth map (the same JSON as `X-Pose`).
    var pose: String? = nil
}

/// Source effectivement utilisée pour le flux.
enum CaptureSource: Equatable, Sendable {
    /// `lens`: the lens actually in use (1× when the ultra wide one was asked for but is missing).
    /// `depth`: the LiDAR depth is captured with the video.
    case camera(name: String, lens: CameraLens, depth: Bool)
    /// The back camera through ARKit, each picture with the camera's pose (`ARPoseCapture`).
    case arkit
    /// Aucune caméra (simulateur) : une mire animée est diffusée à la place.
    case testPattern
    case failed(String)
}

/// Capture la caméra et encode les images en JPEG.
/// La session est manipulée sur `sessionQueue`, l'encodage a lieu sur `videoQueue`.
final class CameraManager: NSObject, @unchecked Sendable {
    let session = AVCaptureSession()

    /// Indique si quelqu'un regarde le flux. Sans spectateur, on n'encode qu'une image
    /// par seconde (pour `/snapshot.jpg`) afin d'économiser la batterie.
    var hasViewers: @Sendable () -> Bool = { false }
    var onFrame: (@Sendable (Data) -> Void)?
    var onFrameSizeChange: (@Sendable (CGSize) -> Void)?

    private struct EncoderSettings: Sendable {
        var maxFPS = 30
        var quality: CGFloat = 0.6
    }

    private let sessionQueue = DispatchQueue(label: "camstream.session")
    private let videoQueue = DispatchQueue(label: "camstream.video", qos: .userInitiated)
    private let depthQueue = DispatchQueue(label: "camstream.depth", qos: .userInitiated)
    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    /// The newest depth map, at most `depthPeriod` old while the LiDAR runs.
    private let depth = OSAllocatedUnfairLock<DepthFrame?>(initialState: nil)
    private let videoRotation = OSAllocatedUnfairLock<Double>(initialState: 0)
    /// Name of the stabilization mode asked of the video connection ("off" when none).
    private let stabilizationName = OSAllocatedUnfairLock<String>(initialState: "off")
    /// Depth maps are converted at most this often (s): ~5 a second is plenty for a map.
    private let depthPeriod: CFTimeInterval = 0.2
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private let settings = OSAllocatedUnfairLock(initialState: EncoderSettings())

    // sessionQueue uniquement
    private var videoInput: AVCaptureDeviceInput?
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private var rotationObservation: NSKeyValueObservation?
    private var testPatternTimer: DispatchSourceTimer?

    // videoQueue uniquement
    private var nextFrameTime: CFTimeInterval = 0
    private var lastFrameSize: CGSize = .zero

    // depthQueue uniquement
    private var lastDepthTime: CFTimeInterval = 0

    func updateEncoder(fps: Int, quality: JPEGQuality) {
        settings.withLock { $0 = EncoderSettings(maxFPS: fps, quality: quality.compression) }
    }

    /// The physical camera of this type at this position, or nil when the iPhone has none.
    static func camera(_ type: AVCaptureDevice.DeviceType,
                       position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        AVCaptureDevice.DiscoverySession(deviceTypes: [type], mediaType: .video, position: position)
            .devices.first
    }

    /// True when the back camera has an ultra wide lens (iPhone 11 and later, not the SE).
    static var hasUltraWideCamera: Bool {
        camera(.builtInUltraWideCamera, position: .back) != nil
    }

    /// True when the iPhone has a LiDAR scanner (Pro models from the iPhone 12 Pro).
    static var hasLiDAR: Bool {
        camera(.builtInLiDARDepthCamera, position: .back) != nil
    }

    /// The stabilization mode in use: "off", "standard", "cinematic", "cinematic-extended" or
    /// "cinematic-extended-enhanced".
    func stabilizationInUse() -> String {
        stabilizationName.withLock { $0 }
    }

    /// The newest LiDAR depth map, or nil (LiDAR off, missing, or nothing received yet).
    func latestDepth() -> DepthFrame? {
        depth.withLock { $0 }
    }

    /// The newest depth map as a grey JPEG for a quick look: near is white, 5 m and beyond
    /// black, no measure black. In the sensor's own orientation, like `/depth`.
    func depthPreviewJPEG(of given: DepthFrame? = nil) -> Data? {
        guard let frame = given ?? latestDepth() else { return nil }
        var grey = [UInt8](repeating: 0, count: frame.width * frame.height)
        frame.metres.withUnsafeBytes { raw in
            let metres = raw.bindMemory(to: Float32.self)
            for i in 0..<min(grey.count, metres.count) {
                let d = metres[i]
                if d.isFinite && d > 0 {
                    grey[i] = UInt8(max(0, min(255, 255 * (1 - d / 5))))
                }
            }
        }
        let image = CIImage(bitmapData: Data(grey), bytesPerRow: frame.width,
                            size: CGSize(width: frame.width, height: frame.height),
                            format: .L8, colorSpace: nil)
        let key = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        return ciContext.jpegRepresentation(of: image, colorSpace: colorSpace, options: [key: 0.8])
    }

    /// Configure (ou reconfigure) la capture puis la démarre.
    /// `lens` only applies to the back camera; the front one is always 1×. `lidar`: capture the
    /// LiDAR depth too (back camera of a Pro iPhone); the video is then the 1× lens's.
    /// `stabilization`: none while the LiDAR depth is captured.
    func start(position: AVCaptureDevice.Position, lens: CameraLens, lidar: Bool,
               resolution: StreamResolution, stabilization: VideoStabilization,
               completion: @escaping @Sendable (CaptureSource) -> Void) {
        sessionQueue.async { [self] in
            let source = configure(position: position, lens: lens, lidar: lidar,
                                   resolution: resolution, stabilization: stabilization)
            switch source {
            case .camera:
                stopTestPattern()
                if !session.isRunning { session.startRunning() }
            case .testPattern:
                if session.isRunning { session.stopRunning() }
                startTestPattern(resolution: resolution)
            case .failed, .arkit:
                break
            }
            completion(source)
        }
    }

    /// `completion` runs once the camera is released (ARKit can then take it).
    func stop(completion: (@Sendable () -> Void)? = nil) {
        sessionQueue.async { [self] in
            if session.isRunning { session.stopRunning() }
            stopTestPattern()
            depth.withLock { $0 = nil } // no stale depth once the camera is let go
            completion?()
        }
    }

    // MARK: - Configuration

    private func configure(position: AVCaptureDevice.Position, lens: CameraLens, lidar: Bool,
                           resolution: StreamResolution,
                           stabilization: VideoStabilization) -> CaptureSource {
        stabilizationName.withLock { $0 = "off" } // until set on the new connection below
        let wanted: CameraLens = position == .back ? lens : .wide
        // The LiDAR camera is the 1× lens with the LiDAR scanner: it wins over the lens choice.
        let lidarDevice = lidar && position == .back
            ? Self.camera(.builtInLiDARDepthCamera, position: .back) : nil
        guard let device = lidarDevice
                ?? Self.camera(wanted.deviceType, position: position)
                ?? Self.camera(.builtInWideAngleCamera, position: position)
                ?? AVCaptureDevice.default(for: .video)
        else {
            return .testPattern
        }
        let usedLens: CameraLens = device.deviceType == .builtInUltraWideCamera ? .ultraWide : .wide

        session.beginConfiguration()

        if let videoInput {
            session.removeInput(videoInput)
            self.videoInput = nil
        }
        if session.outputs.contains(depthOutput) {
            session.removeOutput(depthOutput)
        }
        depth.withLock { $0 = nil }
        // The preset chosen for the previous camera may not exist on this one (other lens):
        // a preset every camera has, the chosen one is set again below.
        session.sessionPreset = .high

        let input: AVCaptureDeviceInput
        do {
            input = try AVCaptureDeviceInput(device: device)
        } catch {
            session.commitConfiguration()
            return .failed("Caméra inaccessible : \(error.localizedDescription)")
        }
        guard session.canAddInput(input) else {
            session.commitConfiguration()
            return .failed("Impossible d'utiliser cette caméra.")
        }
        session.addInput(input)
        videoInput = input

        // With the LiDAR, a format that has depth is chosen on the device itself (the session
        // preset then becomes .inputPriority); without one, the usual preset and no depth.
        let depthFormatSet = lidarDevice != nil && selectDepthFormat(device, resolution: resolution)
        if !depthFormatSet {
            session.sessionPreset = session.canSetSessionPreset(resolution.preset) ? resolution.preset : .high
        }

        if !session.outputs.contains(videoOutput) {
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
            }
        }

        var capturesDepth = false
        if depthFormatSet, session.canAddOutput(depthOutput) {
            depthOutput.isFilteringEnabled = true // holes filled, edges smoothed
            depthOutput.alwaysDiscardsLateDepthData = true
            depthOutput.setDelegate(self, callbackQueue: depthQueue)
            session.addOutput(depthOutput)
            capturesDepth = true
        }

        session.commitConfiguration()

        setupRotation(for: device)
        applyStabilization(capturesDepth ? .off : stabilization, device: device)
        return .camera(name: device.localizedName, lens: usedLens, depth: capturesDepth)
    }

    /// Asks the video connection for the strongest mode of `wanted` that the camera's active
    /// format supports (none when it supports none of them).
    private func applyStabilization(_ wanted: VideoStabilization, device: AVCaptureDevice) {
        var used: VideoStabilization.Choice?
        if let connection = videoOutput.connection(with: .video),
           connection.isVideoStabilizationSupported {
            used = wanted.choices.first { device.activeFormat.isVideoStabilizationModeSupported($0.mode) }
            connection.preferredVideoStabilizationMode = used?.mode ?? .off
        }
        let name = used?.name ?? "off"
        stabilizationName.withLock { $0 = name }
    }

    /// Picks, on the LiDAR camera, the video format nearest to `resolution` that also gives
    /// depth, and its largest Float32 (else Float16) depth format. False when none fits.
    private func selectDepthFormat(_ device: AVCaptureDevice, resolution: StreamResolution) -> Bool {
        let width = Int32(resolution.testPatternSize.width)
        let withDepth = device.formats.filter { !$0.supportedDepthDataFormats.isEmpty && !$0.isVideoBinned }
        guard let format = withDepth.min(by: {
            abs($0.formatDescription.dimensions.width - width)
                < abs($1.formatDescription.dimensions.width - width)
        }) else { return false }
        let depthFormats = format.supportedDepthDataFormats
        func depthFormat(_ type: OSType) -> AVCaptureDevice.Format? {
            depthFormats
                .filter { $0.formatDescription.mediaSubType.rawValue == type }
                .max { $0.formatDescription.dimensions.width < $1.formatDescription.dimensions.width }
        }
        guard let chosen = depthFormat(kCVPixelFormatType_DepthFloat32)
                ?? depthFormat(kCVPixelFormatType_DepthFloat16)
                ?? depthFormats.last
        else { return false }
        do {
            try device.lockForConfiguration()
            device.activeFormat = format
            device.activeDepthDataFormat = chosen
            device.unlockForConfiguration()
            return true
        } catch {
            return false
        }
    }

    /// Keeps the newest depth map (converted to Float32 metres), at most every `depthPeriod`.
    private func store(_ depthData: AVDepthData) {
        let now = CACurrentMediaTime()
        guard now - lastDepthTime >= depthPeriod else { return }
        lastDepthTime = now

        let metric = depthData.depthDataType == kCVPixelFormatType_DepthFloat32
            ? depthData : depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let map = metric.depthDataMap
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return }
        let width = CVPixelBufferGetWidth(map), height = CVPixelBufferGetHeight(map)
        let rowBytes = CVPixelBufferGetBytesPerRow(map)
        var metres = Data(count: width * height * MemoryLayout<Float32>.size)
        metres.withUnsafeMutableBytes { (out: UnsafeMutableRawBufferPointer) in
            guard let start = out.baseAddress else { return }
            for row in 0..<height {
                memcpy(start + row * width * MemoryLayout<Float32>.size, base + row * rowBytes,
                       width * MemoryLayout<Float32>.size)
            }
        }

        var intrinsics: [Double]?
        if let calibration = metric.cameraCalibrationData {
            let k = calibration.intrinsicMatrix // relative to intrinsicMatrixReferenceDimensions
            let reference = calibration.intrinsicMatrixReferenceDimensions
            if reference.width > 0, reference.height > 0 {
                let sx = Double(width) / Double(reference.width)
                let sy = Double(height) / Double(reference.height)
                intrinsics = [Double(k.columns.0.x) * sx, Double(k.columns.1.y) * sy,
                              Double(k.columns.2.x) * sx, Double(k.columns.2.y) * sy]
            }
        }
        let frame = DepthFrame(width: width, height: height, metres: metres, intrinsics: intrinsics,
                               videoRotation: videoRotation.withLock { $0 }, time: now)
        depth.withLock { $0 = frame }
    }

    /// L'image diffusée suit l'orientation physique de l'iPhone (portrait ou paysage),
    /// même si l'interface reste en portrait.
    private func setupRotation(for device: AVCaptureDevice) {
        rotationObservation = nil
        let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: nil)
        rotationCoordinator = coordinator
        applyCaptureRotation(coordinator.videoRotationAngleForHorizonLevelCapture)

        rotationObservation = coordinator.observe(\.videoRotationAngleForHorizonLevelCapture, options: [.new]) { [weak self] _, change in
            guard let self, let angle = change.newValue else { return }
            self.sessionQueue.async { self.applyCaptureRotation(angle) }
        }
    }

    private func applyCaptureRotation(_ angle: CGFloat) {
        guard let connection = videoOutput.connection(with: .video),
              connection.isVideoRotationAngleSupported(angle)
        else { return }
        connection.videoRotationAngle = angle
        videoRotation.withLock { $0 = Double(angle) }
    }

    // MARK: - Mire de test (simulateur / appareil sans caméra)

    private func startTestPattern(resolution: StreamResolution) {
        stopTestPattern()
        let size = resolution.testPatternSize
        let startTime = CACurrentMediaTime()
        let timer = DispatchSource.makeTimerSource(queue: videoQueue)
        timer.schedule(deadline: .now(), repeating: .milliseconds(33))
        timer.setEventHandler { [weak self] in
            guard let self, self.shouldEncodeFrame() else { return }
            let image = TestPattern.render(size: size, elapsed: CACurrentMediaTime() - startTime)
            self.encode(image)
        }
        testPatternTimer = timer
        timer.resume()
    }

    private func stopTestPattern() {
        testPatternTimer?.cancel()
        testPatternTimer = nil
    }

    // MARK: - Encodage (videoQueue)

    /// Limite la cadence d'encodage au nombre d'images/s choisi.
    private func shouldEncodeFrame() -> Bool {
        let now = CACurrentMediaTime()
        let fps = hasViewers() ? Double(settings.withLock { $0.maxFPS }) : 1
        let interval = 1 / max(fps, 1)
        // Petite tolérance pour absorber la gigue de la caméra.
        guard now + 0.004 >= nextFrameTime else { return false }
        nextFrameTime = max(nextFrameTime + interval, now)
        return true
    }

    private func encode(_ image: CIImage) {
        let quality = settings.withLock { $0.quality }
        let key = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        guard let jpeg = ciContext.jpegRepresentation(of: image, colorSpace: colorSpace, options: [key: quality]) else {
            return
        }

        let size = image.extent.size
        if size != lastFrameSize {
            lastFrameSize = size
            onFrameSizeChange?(size)
        }
        onFrame?(jpeg)
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard shouldEncodeFrame(), let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        encode(CIImage(cvPixelBuffer: pixelBuffer))
    }
}

extension CameraManager: AVCaptureDepthDataOutputDelegate {
    func depthDataOutput(_ output: AVCaptureDepthDataOutput, didOutput depthData: AVDepthData,
                         timestamp: CMTime, connection: AVCaptureConnection) {
        store(depthData)
    }
}
