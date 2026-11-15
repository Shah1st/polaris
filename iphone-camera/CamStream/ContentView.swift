import AVFoundation
import SwiftUI

struct ContentView: View {
    @Bindable var model: StreamModel
    @Environment(\.scenePhase) private var scenePhase
    @State private var showSettings = false
    @State private var isDimmed = false
    @State private var copiedURL: String?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            preview
                .ignoresSafeArea()

            VStack(spacing: 12) {
                topBar
                Spacer()
                bottomPanel
            }
            .padding()

            if isDimmed {
                dimOverlay
            }
        }
        .statusBarHidden(isDimmed)
        .persistentSystemOverlays(isDimmed ? .hidden : .automatic)
        .sheet(isPresented: $showSettings) {
            SettingsView(model: model)
        }
        .task {
            await model.start()
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { model.sceneBecameActive() }
        }
    }

    // MARK: Aperçu

    @ViewBuilder
    private var preview: some View {
        switch model.captureSource {
        case .camera:
            CameraPreview(session: model.camera.session, configurationID: model.configurationID)
        case .arkit:
            placeholder(
                icon: "move.3d",
                title: "Position suivie (ARKit)",
                message: "Chaque image part avec la position de la caméra. Pas d'aperçu ici : regardez le flux sur l'ordinateur. Au début, bougez doucement l'iPhone au-dessus du sol pour qu'il le trouve."
            )
        case .testPattern:
            placeholder(
                icon: "square.stack.3d.down.right",
                title: "Mire de test",
                message: "Aucune caméra détectée (simulateur ?). Une mire animée est diffusée à la place pour tester le réseau."
            )
        case .failed(let message):
            placeholder(icon: "exclamationmark.triangle", title: "Caméra indisponible", message: message)
        case nil:
            if model.cameraAccessDenied {
                cameraDeniedView
            } else if model.isRunning {
                ProgressView().tint(.white)
            } else {
                placeholder(
                    icon: "video.slash",
                    title: "Diffusion arrêtée",
                    message: "Touchez « Démarrer » pour rendre la caméra disponible sur le réseau local."
                )
            }
        }
    }

    private func placeholder(icon: String, title: String, message: String) -> some View {
        ContentUnavailableView {
            Label(title, systemImage: icon)
        } description: {
            Text(message)
        }
        .foregroundStyle(.white)
    }

    private var cameraDeniedView: some View {
        ContentUnavailableView {
            Label("Accès à la caméra refusé", systemImage: "camera.badge.ellipsis")
        } description: {
            Text("Autorisez CamStream à utiliser la caméra dans Réglages pour diffuser son image.")
        } actions: {
            Button("Ouvrir Réglages") {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            .buttonStyle(.borderedProminent)
        }
        .foregroundStyle(.white)
    }

    // MARK: Barre du haut

    private var topBar: some View {
        HStack(spacing: 8) {
            HStack(spacing: 6) {
                Circle()
                    .fill(model.isLive ? Color.red : Color.gray)
                    .frame(width: 8, height: 8)
                Text(model.statusText)
                    .lineLimit(1)
            }
            .font(.footnote.weight(.semibold))
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial, in: Capsule())

            if model.isLive {
                Label("\(model.viewerCount)", systemImage: "eye.fill")
                    .font(.footnote.weight(.semibold).monospacedDigit())
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: Capsule())
                    .accessibilityLabel("\(model.viewerCount) spectateur(s)")
            }

            Spacer()

            Button {
                withAnimation { isDimmed = true }
            } label: {
                Image(systemName: "moon.fill")
                    .frame(width: 38, height: 38)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .accessibilityLabel("Éteindre l'écran (la diffusion continue)")
        }
        .foregroundStyle(.white)
    }

    // MARK: Panneau du bas

    private var bottomPanel: some View {
        VStack(alignment: .leading, spacing: 14) {
            connectionInfo

            HStack(spacing: 12) {
                roundButton(icon: "arrow.triangle.2.circlepath.camera", label: "Changer de caméra") {
                    model.switchCamera()
                }
                .disabled(model.captureSource == .testPattern)

                if model.ultraWideAvailable {
                    lensButton
                }

                roundButton(icon: "slider.horizontal.3", label: "Réglages") {
                    showSettings = true
                }

                Spacer()

                Button {
                    model.toggle()
                } label: {
                    Label(model.isRunning ? "Arrêter" : "Démarrer",
                          systemImage: model.isRunning ? "stop.fill" : "play.fill")
                        .font(.headline)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 4)
                }
                .buttonStyle(.borderedProminent)
                .tint(model.isRunning ? .red : .green)
            }
        }
        .padding(16)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
    }

    @ViewBuilder
    private var connectionInfo: some View {
        if model.isRunning, model.addresses.isEmpty {
            Label("Connectez l'iPhone au même réseau Wi-Fi que votre Mac.", systemImage: "wifi.exclamationmark")
                .font(.subheadline)
        } else if model.isLive, let url = model.primaryURL {
            VStack(alignment: .leading, spacing: 6) {
                Text("Ouvrez cette adresse sur votre Mac")
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                HStack(spacing: 10) {
                    Text(url)
                        .font(.system(.title3, design: .monospaced).weight(.semibold))
                        .lineLimit(1)
                        .minimumScaleFactor(0.5)
                        .textSelection(.enabled)
                    Spacer(minLength: 0)
                    Button {
                        copy(url)
                    } label: {
                        Image(systemName: copiedURL == url ? "checkmark" : "doc.on.doc")
                            .frame(width: 32, height: 32)
                    }
                    .accessibilityLabel("Copier l'adresse")
                }

                Text("Flux brut (VLC, OBS…) : \(url)/stream")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)
            }
        } else if model.isRunning {
            Text("Démarrage du serveur…")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        } else {
            Text("La caméra n'est pas partagée.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    /// 1× / 0,5× like Apple's Camera app, yellow on the ultra wide. The front camera is always 1×.
    private var lensButton: some View {
        // the front camera and the LiDAR camera are always 1×
        let shown: CameraLens = model.position == .back && !model.lidar ? model.lens : .wide
        let ultraWide = shown == .ultraWide
        return Button {
            model.switchLens()
        } label: {
            Text(shown.label)
                .font(.subheadline.weight(.bold))
                .frame(width: 46, height: 46)
                .background(Color.white.opacity(0.12), in: Circle())
        }
        .foregroundStyle(ultraWide ? Color.yellow : Color.white)
        .disabled(!model.canChooseLens)
        .accessibilityLabel(ultraWide ? "Objectif grand angle 0,5×, passer en 1×" : "Objectif 1×, passer en grand angle 0,5×")
    }

    private func roundButton(icon: String, label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.title3)
                .frame(width: 46, height: 46)
                .background(Color.white.opacity(0.12), in: Circle())
        }
        .foregroundStyle(.white)
        .accessibilityLabel(label)
    }

    private func copy(_ url: String) {
        UIPasteboard.general.string = url
        copiedURL = url
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            if copiedURL == url { copiedURL = nil }
        }
    }

    // MARK: Écran éteint

    private var dimOverlay: some View {
        Color.black
            .ignoresSafeArea()
            .overlay {
                VStack(spacing: 10) {
                    Image(systemName: "dot.radiowaves.left.and.right")
                        .font(.largeTitle)
                    Text(model.isLive ? "Diffusion en cours · \(model.viewerCount) spectateur(s)" : model.statusText)
                    Text("Touchez l'écran pour réafficher")
                        .font(.footnote)
                }
                .foregroundStyle(.white.opacity(0.3))
            }
            .contentShape(Rectangle())
            .onTapGesture {
                withAnimation { isDimmed = false }
            }
            .transition(.opacity)
    }
}
