import Foundation

enum NetworkInfo {
    /// Adresses IPv4 locales utilisables par les autres appareils du réseau.
    /// `en*` = Wi-Fi / Ethernet, `bridge*` = partage de connexion de l'iPhone.
    static func localIPv4Addresses() -> [String] {
        var ifaddrPointer: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddrPointer) == 0, let first = ifaddrPointer else { return [] }
        defer { freeifaddrs(ifaddrPointer) }

        var wifi: [String] = []
        var others: [String] = []

        for pointer in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let interface = pointer.pointee
            let flags = Int32(interface.ifa_flags)
            let isUp = (flags & IFF_UP) != 0 && (flags & IFF_RUNNING) != 0
            let isLoopback = (flags & IFF_LOOPBACK) != 0
            guard isUp, !isLoopback,
                  let address = interface.ifa_addr,
                  address.pointee.sa_family == UInt8(AF_INET)
            else { continue }

            let name = String(cString: interface.ifa_name)
            guard name.hasPrefix("en") || name.hasPrefix("bridge") else { continue }

            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            let result = getnameinfo(address, socklen_t(address.pointee.sa_len),
                                     &host, socklen_t(host.count),
                                     nil, 0, NI_NUMERICHOST)
            guard result == 0 else { continue }
            let ip = host.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }

            // On ignore les adresses auto-attribuées (pas de DHCP).
            guard !ip.hasPrefix("169.254.") else { continue }

            if name == "en0" { wifi.append(ip) } else { others.append(ip) }
        }
        return wifi + others
    }
}
