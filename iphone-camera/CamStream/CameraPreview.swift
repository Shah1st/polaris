import AVFoundation
import SwiftUI

/// Aperçu local de la caméra (ce que voient les spectateurs).
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    /// Change à chaque reconfiguration pour réappliquer la rotation de l'aperçu.
    let configurationID: Int

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.backgroundColor = .black
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspect
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {
        view.applyPortraitRotation()
    }

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        var previewLayer: AVCaptureVideoPreviewLayer {
            // swiftlint:disable:next force_cast
            layer as! AVCaptureVideoPreviewLayer
        }

        /// L'interface est verrouillée en portrait : l'aperçu est donc toujours tourné de 90°,
        /// comme dans l'app Appareil photo. Le flux diffusé, lui, suit l'orientation réelle.
        func applyPortraitRotation() {
            guard let connection = previewLayer.connection,
                  connection.isVideoRotationAngleSupported(90),
                  connection.videoRotationAngle != 90
            else { return }
            connection.videoRotationAngle = 90
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            applyPortraitRotation()
        }
    }
}
