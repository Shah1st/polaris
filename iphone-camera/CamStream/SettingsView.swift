import AVFoundation
import SwiftUI

struct SettingsView: View {
    @Bindable var model: StreamModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Caméra") {
                    Picker("Caméra", selection: $model.position) {
                        Text("Arrière").tag(AVCaptureDevice.Position.back)
                        Text("Avant").tag(AVCaptureDevice.Position.front)
                    }
                    .pickerStyle(.segmented)
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
