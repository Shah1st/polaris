import AVFoundation
import CoreImage
import ImageIO
import QuartzCore
import os

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

/// Source effectivement utilisée pour le flux.
enum CaptureSource: Equatable, Sendable {
    /// `lens`: the lens actually in use (1× when the ultra wide one was asked for but is missing).
    case camera(name: String, lens: CameraLens)
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
    private let videoOutput = AVCaptureVideoDataOutput()
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

    /// Configure (ou reconfigure) la capture puis la démarre.
    /// `lens` only applies to the back camera; the front one is always 1×.
    func start(position: AVCaptureDevice.Position, lens: CameraLens, resolution: StreamResolution,
               completion: @escaping @Sendable (CaptureSource) -> Void) {
        sessionQueue.async { [self] in
            let source = configure(position: position, lens: lens, resolution: resolution)
            switch source {
            case .camera:
                stopTestPattern()
                if !session.isRunning { session.startRunning() }
            case .testPattern:
                if session.isRunning { session.stopRunning() }
                startTestPattern(resolution: resolution)
            case .failed:
                break
            }
            completion(source)
        }
    }

    func stop() {
        sessionQueue.async { [self] in
            if session.isRunning { session.stopRunning() }
            stopTestPattern()
        }
    }

    // MARK: - Configuration

    private func configure(position: AVCaptureDevice.Position, lens: CameraLens,
                           resolution: StreamResolution) -> CaptureSource {
        let wanted: CameraLens = position == .back ? lens : .wide
        guard let device = Self.camera(wanted.deviceType, position: position)
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

        session.sessionPreset = session.canSetSessionPreset(resolution.preset) ? resolution.preset : .high

        if !session.outputs.contains(videoOutput) {
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
            }
        }

        session.commitConfiguration()

        setupRotation(for: device)
        return .camera(name: device.localizedName, lens: usedLens)
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
