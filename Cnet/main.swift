// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

print("Hello, World!")
//swiftlint:disable all

enum Config {
    static let singleInputSize = CNetIOSize(channels: 1, height: 6, width: 7)
    static let kernelSize = CNetIOSize(channels: 4, height: 6, width: 7)

    static let singleConvolutionOutputSize =
        CNetIOSize(channels: 4, height: 6, width: 7)

    static let fullConvolutionOutputSize = CNetIOSize(
        channels: 4, height: 6, width: singleConvolutionOutputSize.width * 2
    )

    static let topFcInputSize =
        CNetIOSize(vectorWidth: fullConvolutionOutputSize.volume)

    static let finalOutputSize = CNetIOSize(vectorWidth: 42)
}

enum World {
    static let device = MTLCopyAllDevices()[0]

    static let gridInputs = UnsafeMutableBufferPointer<FF32>.allocate(
        capacity: 2 * Config.singleInputSize.volume
    )

    static let finalOutputs = UnsafeMutableBufferPointer<FF32>.allocate(
        capacity: Config.finalOutputSize.volume
    )

    static let player0 = World.kernelWeights.withUnsafeBufferPointer {
        CWConvolution(
            World.device,
            destinationIOSize: Config.singleConvolutionOutputSize,
            kernelIOSize: Config.kernelSize,
            sourceIOSize: Config.singleInputSize,
            kernelWeights: $0
        )
    }

    static let player1 = World.kernelWeights.withUnsafeBufferPointer {
        CWConvolution(
            World.device,
            destinationIOSize: Config.singleConvolutionOutputSize,
            kernelIOSize: Config.kernelSize,
            sourceIOSize: Config.singleInputSize,
            kernelWeights: $0
        )
    }

    static let topFc = CFullyConnected(
        World.device, destinationIOSize: Config.finalOutputSize,
        sourceIOSize: Config.topFcInputSize
    )

    static let pStatics = UnsafeMutableBufferPointer<FF32>.allocate(
        capacity: topFc.cBiases + topFc.cWeights
    )
}

let net = CNet(
    World.device, convolution0: World.player0, convolution1: World.player1,
    fullyConnected: World.topFc, input: World.gridInputs,
    output: World.finalOutputs
)

let cb = World.topFc.cBiases
var pBiases = UnsafeMutableBufferPointer(rebasing: World.pStatics[..<cb])
var pWeights = UnsafeMutableBufferPointer(rebasing: World.pStatics[cb...])

pBiases.initialize(repeating: 0)
pWeights.initialize(repeating: 1)

net.setStaticsBuffer(World.pStatics)

World.gridInputs.initialize(repeating: 1)
World.finalOutputs.initialize(repeating: 42.42)

net.activate()

print(World.finalOutputs.map { $0 })
