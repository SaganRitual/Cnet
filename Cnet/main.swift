// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

print("Hello, World!")
//swiftlint:disable all

enum Config {
    static let winLength = 4
    static let inputImageWidth = 7
    static let inputImageHeight = 6
    static let inputImageArea = 42

    static let diagonalKernelWidth = 4
    static let diagonalKernelHeight = 4
    static let diagonalKernelArea = 16

    static let verticalKernelWidth = 1
    static let verticalKernelHeight = 4
    static let verticalKernelArea = 4

    static let horizontalKernelWidth = 4
    static let horizontalKernelHeight = 1
    static let horizontalKernelArea = 4
}

enum World {
    static let device = MTLCopyAllDevices()[0]
}

class Main {
    init() {
        let horizontalWeights =
            UnsafeMutableBufferPointer<FF32>.allocate(capacity: Config.horizontalKernelArea)

        horizontalWeights.initialize(repeating: 1)

        let verticalWeights =
            UnsafeMutableBufferPointer<FF32>.allocate(capacity: Config.verticalKernelArea)

        verticalWeights.initialize(repeating: 1)

        let negativeSlopeWeights = UnsafeMutableBufferPointer<FF32>.allocate(
            capacity: Config.diagonalKernelArea
        )

        _ = negativeSlopeWeights.initialize(from: [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ])

        let positiveSlopeWeights = UnsafeMutableBufferPointer<FF32>.allocate(
            capacity: Config.diagonalKernelArea
        )

        _ = positiveSlopeWeights.initialize(from: [
            0, 0, 0, 1,
            0, 0, 1, 0,
            0, 1, 0, 0,
            1, 0, 0, 0
        ])

        let horizontalKernel = CConvolution(
            device: World.device, tier: .top,
            imageWidth: Config.inputImageWidth, imageHeight: Config.inputImageHeight,
            kernelWidth: Config.horizontalKernelWidth, kernelHeight: Config.horizontalKernelHeight,
            kernelWeights: UnsafeBufferPointer(horizontalWeights)
        )

        let verticalKernel = CConvolution(
            device: World.device, tier: .top,
            imageWidth: Config.inputImageWidth, imageHeight: Config.inputImageHeight,
            kernelWidth: Config.verticalKernelWidth, kernelHeight: Config.verticalKernelHeight,
            kernelWeights: UnsafeBufferPointer(verticalWeights)
        )

        let negativeSlopeKernel = CConvolution(
            device: World.device, tier: .top,
            imageWidth: Config.inputImageWidth, imageHeight: Config.inputImageHeight,
            kernelWidth: Config.diagonalKernelWidth, kernelHeight: Config.diagonalKernelHeight,
            kernelWeights: UnsafeBufferPointer(negativeSlopeWeights)
        )

        let positiveSlopeKernel = CConvolution(
            device: World.device, tier: .top,
            imageWidth: Config.inputImageWidth, imageHeight: Config.inputImageHeight,
            kernelWidth: Config.diagonalKernelWidth, kernelHeight: Config.diagonalKernelHeight,
            kernelWeights: UnsafeBufferPointer(positiveSlopeWeights)
        )

        let negativeSlopeNet = CNet(World.device, structure: CNetStructure([negativeSlopeKernel]))
        let positiveSlopeNet = CNet(World.device, structure: CNetStructure([positiveSlopeKernel]))
        let verticalNet = CNet(World.device, structure: CNetStructure([verticalKernel]))
        let horizontalNet = CNet(World.device, structure: CNetStructure([horizontalKernel]))

        let input: [FF32] = (0..<Config.inputImageArea).map { _ in 1 }//FF32(Int.random(in: -1...1)) }

        var positiveSlopeOutput = [FF32](repeating: 42, count: negativeSlopeNet.destination.imageArea)
        var negativeSlopeOutput = [FF32](repeating: 42, count: positiveSlopeNet.destination.imageArea)
        var verticalOutput = [FF32](repeating: 42, count: verticalNet.destination.imageArea)
        var horizontalOutput = [FF32](repeating: 42, count: horizontalNet.destination.imageArea)

        positiveSlopeNet.activate(input: input, result: &positiveSlopeOutput)
        negativeSlopeNet.activate(input: input, result: &negativeSlopeOutput)
        verticalNet.activate(input: input, result: &verticalOutput)
        horizontalNet.activate(input: input, result: &horizontalOutput)

        print(input)
        print(positiveSlopeOutput)
        print(negativeSlopeOutput)
        print(verticalOutput)
        print(horizontalOutput)
    }
}

_ = Main()
