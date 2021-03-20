// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

print("Hello, World!")
//swiftlint:disable all

struct CNetIO {
    var area: Int { width * height }
    var volume: Int { area * channels }

    let channels: Int
    let height: Int
    let width: Int
}

enum Config {
    static let winLength = 4

    static let inputSpec =  CNetIO(channels: 1, height: 6, width: 7)
    static let kernelSpec = CNetIO(channels: 4, height: 6, width: 7)
    static let outputSpec = CNetIO(channels: 4, height: 6, width: 7)
}

enum World {
    static let device = MTLCopyAllDevices()[0]
}

class Main {
    init() {
        let kernelWeights: [FF32] = [
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 8, 4, 2, 1,    // Horizontal

            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 8,    // Vertical
            0, 0, 0, 0, 0, 0, 4,    // e
            0, 0, 0, 0, 0, 0, 2,    // r
            0, 0, 0, 0, 0, 0, 1,    // t

            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 8, 0, 0, 0,    // Slope negative
            0, 0, 0, 0, 4, 0, 0,    //  l
            0, 0, 0, 0, 0, 2, 0,    //   o
            0, 0, 0, 0, 0, 0, 1,    //    p

            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 8,    // Slope positive
            0, 0, 0, 0, 0, 4, 0,    //   o
            0, 0, 0, 0, 2, 0, 0,    //  l
            0, 0, 0, 1, 0, 0, 0     // S
        ]

        precondition(
            kernelWeights.count == Config.kernelSpec.volume,
            "kernelWeights.count (\(kernelWeights.count))"
            + " != kernel volume (\(Config.kernelSpec.volume))"
        )

        let kernel = kernelWeights.withUnsafeBufferPointer {
            CConvolution(
                device: World.device, tier: .top,
                destinationIoSpec: Config.outputSpec,
                kernelIoSpec: Config.kernelSpec,
                sourceIoSpec: Config.inputSpec,
                kernelWeights: $0
            )
        }

        let convolveNet = CNet(World.device, structure: CNetStructure([kernel]))

        let input: [FF32] = (0..<Config.inputSpec.volume).map
            { _ in 1 } //FF32(Int.random(in: -1...1)) }

        var output = [FF32](repeating: 42, count: Config.outputSpec.volume)

        convolveNet.activate(input: input, result: &output)

        print(input)
        print(output)
    }
}

_ = Main()
