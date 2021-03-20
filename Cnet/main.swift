// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

print("Hello, World!")
//swiftlint:disable all

struct CNetIOSpec {
    var area: Int { width * height }
    var volume: Int { area * channels }

    let channels: Int
    let height: Int
    let width: Int
}

enum Config {
    static let winLength = 4

    static let inputSpec =  CNetIOSpec(channels: 1, height: 6, width: 7)
    static let kernelSpec = CNetIOSpec(channels: 4, height: 6, width: 7)
    static let outputSpec = CNetIOSpec(channels: 4, height: 6, width: 7)
}

enum World {
    static let device = MTLCopyAllDevices()[0]
}

class Convolver {
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

        let convolveNet = CNet(World.device, structure: [kernel])

        let input: [FF32] = (0..<Config.inputSpec.volume).map
            { _ in 1 } //FF32(Int.random(in: -1...1)) }

        var output = [FF32](repeating: 42, count: Config.outputSpec.volume)

        convolveNet.activate(input: input, result: &output)

        print(input)
        print(output)
    }
}

class Matrixer {
    init() {
        let ss = CNetIOSpec(channels: 1, height: 1, width: 42)
        let dd = CNetIOSpec(channels: 1, height: 1, width: 42)

        let biases: [FF32] = .init(repeating: 2, count: dd.width)
        let inputs: [FF32] = .init(repeating: 1, count: ss.width)
        var outputs: [FF32] = .init(repeating: 42.42, count: dd.width)
        let weights: [FF32] = .init(repeating: 1, count: ss.width * dd.width)

        let fc = CFullyConnected(
            World.device, source: ss, destination: dd,
            weightsArray: weights, biasesArray: biases
        )

        let net = CNet(World.device, structure: [fc])

        net.activate(input: inputs, result: &outputs)

        print(inputs)
        print(outputs)
    }
}

_ = Matrixer()
