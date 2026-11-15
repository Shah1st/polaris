import CoreImage
import UIKit

/// Mire animée diffusée quand aucune caméra n'est disponible (ex. simulateur),
/// pour pouvoir tester le flux réseau.
enum TestPattern {
    private static let barColors: [UIColor] = [
        UIColor(white: 0.75, alpha: 1),
        UIColor(red: 0.75, green: 0.75, blue: 0, alpha: 1),
        UIColor(red: 0, green: 0.75, blue: 0.75, alpha: 1),
        UIColor(red: 0, green: 0.75, blue: 0, alpha: 1),
        UIColor(red: 0.75, green: 0, blue: 0.75, alpha: 1),
        UIColor(red: 0.75, green: 0, blue: 0, alpha: 1),
        UIColor(red: 0, green: 0, blue: 0.75, alpha: 1),
    ]

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SS"
        return formatter
    }()

    static func render(size: CGSize, elapsed: TimeInterval) -> CIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true

        let image = UIGraphicsImageRenderer(size: size, format: format).image { context in
            let cg = context.cgContext
            let barWidth = size.width / CGFloat(barColors.count)
            for (index, color) in barColors.enumerated() {
                color.setFill()
                cg.fill(CGRect(x: CGFloat(index) * barWidth, y: 0, width: barWidth + 1, height: size.height))
            }

            // Barre blanche animée pour visualiser la fluidité.
            let sweep = CGFloat(elapsed.truncatingRemainder(dividingBy: 2) / 2)
            UIColor.white.setFill()
            cg.fill(CGRect(x: sweep * (size.width - 12), y: 0, width: 12, height: size.height))

            // Bandeau d'informations.
            let bandHeight = size.height * 0.22
            UIColor(white: 0, alpha: 0.75).setFill()
            cg.fill(CGRect(x: 0, y: (size.height - bandHeight) / 2, width: size.width, height: bandHeight))

            let title = "CamStream · mire de test"
            let time = timeFormatter.string(from: Date())
            let paragraph = NSMutableParagraphStyle()
            paragraph.alignment = .center

            let titleAttributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: size.height * 0.06, weight: .semibold),
                .foregroundColor: UIColor.white,
                .paragraphStyle: paragraph,
            ]
            let timeAttributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.monospacedDigitSystemFont(ofSize: size.height * 0.07, weight: .bold),
                .foregroundColor: UIColor.systemYellow,
                .paragraphStyle: paragraph,
            ]
            let top = (size.height - bandHeight) / 2 + bandHeight * 0.1
            title.draw(in: CGRect(x: 0, y: top, width: size.width, height: bandHeight * 0.4), withAttributes: titleAttributes)
            time.draw(in: CGRect(x: 0, y: top + bandHeight * 0.4, width: size.width, height: bandHeight * 0.5), withAttributes: timeAttributes)
        }

        return CIImage(image: image) ?? CIImage(color: .black).cropped(to: CGRect(origin: .zero, size: size))
    }
}
