import SwiftUI

@main
struct CamStreamApp: App {
    @State private var model = StreamModel()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .preferredColorScheme(.dark)
        }
    }
}
