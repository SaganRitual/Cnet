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

let gridInputs = UnsafeMutableBufferPointer<FF32>.allocate(capacity: Config.inputSpec.volume)
let gridOutputs = UnsafeMutableBufferPointer<FF32>.allocate(capacity: Config.outputSpec.volume)

let layer = World.kernelWeights.withUnsafeBufferPointer {
    CWConvolution(
        device: World.device, tier: .top, destinationIOSize: Config.outputSpec,
        kernelIOSize: Config.kernelSpec, sourceIOSize: Config.inputSpec,
        kernelWeights: $0
    )
}

let net = CNet(
    World.device, structure: [layer], input: gridInputs, output: gridOutputs
)

gridInputs.initialize(repeating: 1)
gridOutputs.initialize(repeating: 42.42)

net.activate()

print(gridOutputs.map { $0 })
