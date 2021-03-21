// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

print("Hello, World!")
//swiftlint:disable all

enum Config {
    static let inputSpec = CNetIOSize(channels: 1, height: 6, width: 7)
    static let kernelSpec = CNetIOSize(channels: 4, height: 6, width: 7)
    static let outputSpec = CNetIOSize(channels: 4, height: 6, width: 7)
}

enum World {
    static let device = MTLCopyAllDevices()[0]
}

let gridInputs = UnsafeMutableBufferPointer<FF32>.allocate(capacity: 2 * Config.inputSpec.volume)
let convolutionOutputs = UnsafeMutableBufferPointer<FF32>.allocate(capacity: 2 * Config.outputSpec.volume)

let player0 = World.kernelWeights.withUnsafeBufferPointer {
    CWConvolution(
        device: World.device, tier: .top, destinationIOSize: Config.outputSpec,
        kernelIOSize: Config.kernelSpec, sourceIOSize: Config.inputSpec,
        kernelWeights: $0
    )
}

let player1 = World.kernelWeights.withUnsafeBufferPointer {
    CWConvolution(
        device: World.device, tier: .top, destinationIOSize: Config.outputSpec,
        kernelIOSize: Config.kernelSpec, sourceIOSize: Config.inputSpec,
        kernelWeights: $0
    )
}

let net = CNet(
    World.device,
    convolution0: player0, convolution1: player1,
    input: gridInputs, output: convolutionOutputs
)

//for ix in stride(from: 0, to: Config.inputSpec.volume, by: 2) {
//    gridInputs[ix + 0] = 1
//    gridInputs[ix + 1] = 0
//}

gridInputs.initialize(repeating: 1)
convolutionOutputs.initialize(repeating: 42.42)

net.activate()

print(convolutionOutputs.map { $0 })
