import AVFoundation
import SwiftUI

struct SettingsView: View {
    @Bindable var model: StreamModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Caméra", selection: $model.position) {
                        Text("Arrière").tag(AVCaptureDevice.Position.back)
                        Text("Avant").tag(AVCaptureDevice.Position.front)
                    }
                    .pickerStyle(.segmented)
                    Picker("Objectif", selection: $model.lens) {
                        ForEach(CameraLens.allCases) { Text($0.title).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .disabled(!model.canChooseLens)
                } header: {
                    Text("Caméra")
                } footer: {
                    Text(lensFooter)
                }

                Section {
                    Toggle("Profondeur LiDAR", isOn: $model.lidar)
                        .disabled(!model.canUseLiDAR)
                } header: {
                    Text("LiDAR")
                } footer: {
                    Text(lidarFooter)
                }

                Section {
                    Toggle("Position (ARKit)", isOn: $model.arkit)
                        .disabled(!model.arkitAvailable || model.position == .front)
                } header: {
                    Text("Position de la caméra")
                } footer: {
                    Text("Chaque image part avec la position et l'orientation de l'iPhone dans la pièce (en-tête X-Pose, aussi sur /pose) : une carte vue du dessus reste stable quand l'iPhone bouge. L'image est alors celle du capteur (paysage) ; avec le LiDAR activé (iPhone Pro), la profondeur part aussi avec la position ; sinon le grand angle 0,5× si cet iPhone le permet avec ARKit.")
                }

                Section {
                    Picker("Résolution", selection: $model.resolution) {
                        ForEach(StreamResolution.allCases) { Text($0.rawValue).tag($0) }
                    }
                    Picker("Images par seconde", selection: $model.fps) {
                        ForEach(StreamModel.fpsOptions, id: \.self) { Text("\($0)").tag($0) }
                    }
                    Picker("Qualité de l'image", selection: $model.quality) {
                        ForEach(JPEGQuality.allCases) { Text($0.rawValue).tag($0) }
                    }
                } header: {
                    Text("Flux vidéo")
                } footer: {
                    Text("Plus de résolution ou de qualité = plus de débit Wi-Fi et un iPhone qui chauffe davantage. 720p · 30 i/s · qualité moyenne convient dans la plupart des cas.")
                }

                Section {
                    if model.pageURLs.isEmpty {
                        Text("Aucune adresse : l'iPhone n'est connecté à aucun réseau local.")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(model.pageURLs, id: \.self) { base in
                        AddressRow(title: "Page web", url: base)
                        AddressRow(title: "Flux MJPEG", url: base + "/stream")
                        AddressRow(title: "Image seule (OBS)", url: base + "/view")
                        AddressRow(title: "Photo instantanée", url: base + "/snapshot.jpg")
                    }
                } header: {
                    Text("Adresses")
                } footer: {
                    Text("Le flux est visible par tous les appareils du réseau local, sans mot de passe. Évitez de l'utiliser sur un Wi-Fi public.")
                }

                Section("Utilisation sur le Mac") {
                    HelpRow(app: "Safari / Chrome", how: "Ouvrir la page web.")
                    HelpRow(app: "VLC", how: "Fichier › Ouvrir un réseau… puis coller l'adresse du flux MJPEG.")
                    HelpRow(app: "OBS", how: "Source « Navigateur » avec l'adresse « Image seule ». Puis « Démarrer la caméra virtuelle » pour l'utiliser dans Zoom, Meet, FaceTime…")
                    HelpRow(app: "ffmpeg / OpenCV", how: "Lire directement l'adresse du flux MJPEG.")
                }
            }
            .navigationTitle("Réglages")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("OK") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    private var lidarFooter: String {
        if !model.lidarAvailable {
            return "Cet iPhone n'a pas de LiDAR (seuls les modèles Pro en ont un, depuis l'iPhone 12 Pro)."
        }
        if model.position == .front {
            return "Le LiDAR n'existe qu'à l'arrière."
        }
        if model.lidar && model.isLive && !model.depthActive {
            return "Profondeur indisponible avec cette caméra : le flux continue sans."
        }
        if model.arkitActive {
            return "Avec la position ARKit, chaque carte de profondeur part avec la position de la caméra (/depth, en-tête X-Pose), environ 5 fois par seconde : la hauteur de chaque point du sol, en mètres. Les points peu sûrs valent 0. Objectif 1×."
        }
        return "Mesure la distance de chaque point (en mètres, jusqu'à environ 5 m), servie sur /depth ; aperçu sur /depth.jpg. Le flux vidéo passe alors par l'objectif 1× : le grand angle n'est pas disponible avec le LiDAR."
    }

    private var lensFooter: String {
        if model.lidar && model.position == .back && model.lidarAvailable {
            return "Avec le LiDAR, la caméra arrière est en 1×."
        }
        if model.arkitActive {
            if model.lens == .ultraWide && model.arkitLens == .wide {
                return "ARKit ne suit pas la position avec le grand angle sur cet iPhone : l'image reste en 1×."
            }
            return "En mode ARKit, l'objectif choisi est demandé à ARKit (le grand angle seulement si cet iPhone le permet). /status dit lequel est utilisé ; les paramètres de la caméra envoyés avec chaque image suivent l'objectif."
        }
        if model.lidar && model.position == .back {
            return "Avec le LiDAR, la caméra arrière est en 1×."
        }
        if !model.ultraWideAvailable {
            return "Cet iPhone n'a pas d'objectif grand angle : la caméra arrière reste en 1×."
        }
        if model.position == .front {
            return "Le grand angle n'existe que sur la caméra arrière."
        }
        return "Grand angle 0,5× : champ environ deux fois plus large, distorsion corrigée par iOS. Après un changement d'objectif, remesurer l'échelle (mètres par pixel) de la carte vue du dessus."
    }
}

private struct AddressRow: View {
    let title: String
    let url: String
    @State private var copied = false

    var body: some View {
        Button {
            UIPasteboard.general.string = url
            copied = true
            Task {
                try? await Task.sleep(for: .seconds(1.5))
                copied = false
            }
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .foregroundStyle(.primary)
                    Text(url)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                }
                Spacer()
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
                    .foregroundStyle(copied ? .green : .accentColor)
            }
        }
        .accessibilityHint("Copie l'adresse")
    }
}

private struct HelpRow: View {
    let app: String
    let how: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(app).font(.subheadline.weight(.semibold))
            Text(how).font(.footnote).foregroundStyle(.secondary)
        }
    }
}
