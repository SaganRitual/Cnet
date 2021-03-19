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

    static let kernelWidth = 4
    static let kernelHeight = 4
    static let kernelArea = 16

    static let outputImageWidth = 4
    static let outputImageHeight = 3
    static let outputImageArea = 12
}

enum World {
    static let device = MTLCopyAllDevices()[0]
}

class Main {
    init() {
        let h0Weights =
            UnsafeMutableBufferPointer<FF32>.allocate(capacity: Config.kernelArea)

        _ = h0Weights.initialize(from: [
            8, 4, 2, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        ])

        let h1Weights =
            UnsafeMutableBufferPointer<FF32>.allocate(capacity: Config.kernelArea)

        _ = h1Weights.initialize(from: [
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            8, 4, 2, 1
        ])

        let v0Weights =
            UnsafeMutableBufferPointer<FF32>.allocate(capacity: Config.kernelArea)

        _ = v0Weights.initialize(from: [
            8, 0, 0, 0,
            4, 0, 0, 0,
            2, 0, 0, 0,
            1, 0, 0, 0
        ])

        let v1Weights =
            UnsafeMutableBufferPointer<FF32>.allocate(capacity: Config.kernelArea)

        _ = v1Weights.initialize(from: [
            0, 0, 0, 8,
            0, 0, 0, 4,
            0, 0, 0, 2,
            0, 0, 0, 1
        ])

        let negativeSlopeWeights = UnsafeMutableBufferPointer<FF32>.allocate(
            capacity: Config.kernelArea
        )

        _ = negativeSlopeWeights.initialize(from: [
            8, 0, 0, 0,
            0, 4, 0, 0,
            0, 0, 2, 0,
            0, 0, 0, 1
        ])

        let positiveSlopeWeights = UnsafeMutableBufferPointer<FF32>.allocate(
            capacity: Config.kernelArea
        )

        _ = positiveSlopeWeights.initialize(from: [
            0, 0, 0, 1,
            0, 0, 2, 0,
            0, 4, 0, 0,
            8, 0, 0, 0
        ])

        let h0Kernel = CConvolution(
            device: World.device, tier: .top,
            imageWidth: Config.inputImageWidth, imageHeight: Config.inputImageHeight,
            kernelWidth: Config.kernelWidth, kernelHeight: Config.kernelHeight,
            kernelWeights: UnsafeBufferPointer(h0Weights)
        )

        let h1Kernel = CConvolution(
            device: World.device, tier: .top,
            imageWidth: Config.inputImageWidth, imageHeight: Config.inputImageHeight,
            kernelWidth: Config.kernelWidth, kernelHeight: Config.kernelHeight,
            kernelWeights: UnsafeBufferPointer(h1Weights)
        )

        let v0Kernel = CConvolution(
            device: World.device, tier: .top,
            imageWidth: Config.inputImageWidth, imageHeight: Config.inputImageHeight,
            kernelWidth: Config.kernelWidth, kernelHeight: Config.kernelHeight,
            kernelWeights: UnsafeBufferPointer(v0Weights)
        )

        let v1Kernel = CConvolution(
            device: World.device, tier: .top,
            imageWidth: Config.inputImageWidth, imageHeight: Config.inputImageHeight,
            kernelWidth: Config.kernelWidth, kernelHeight: Config.kernelHeight,
            kernelWeights: UnsafeBufferPointer(v1Weights)
        )

        let negativeSlopeKernel = CConvolution(
            device: World.device, tier: .top,
            imageWidth: Config.inputImageWidth, imageHeight: Config.inputImageHeight,
            kernelWidth: Config.kernelWidth, kernelHeight: Config.kernelHeight,
            kernelWeights: UnsafeBufferPointer(negativeSlopeWeights)
        )

        let positiveSlopeKernel = CConvolution(
            device: World.device, tier: .top,
            imageWidth: Config.inputImageWidth, imageHeight: Config.inputImageHeight,
            kernelWidth: Config.kernelWidth, kernelHeight: Config.kernelHeight,
            kernelWeights: UnsafeBufferPointer(positiveSlopeWeights)
        )

        let h0Net = CNet(World.device, structure: CNetStructure([h0Kernel]))
        let h1Net = CNet(World.device, structure: CNetStructure([h1Kernel]))
        let v0Net = CNet(World.device, structure: CNetStructure([v0Kernel]))
        let v1Net = CNet(World.device, structure: CNetStructure([v1Kernel]))
        let negativeSlopeNet = CNet(World.device, structure: CNetStructure([negativeSlopeKernel]))
        let positiveSlopeNet = CNet(World.device, structure: CNetStructure([positiveSlopeKernel]))

        let input: [FF32] = (0..<Config.inputImageArea).map { _ in FF32(Int.random(in: -1...1)) }

        var h0Output = [FF32](repeating: 42, count: h0Net.destination.imageArea)
        var h1Output = [FF32](repeating: 42, count: h1Net.destination.imageArea)
        var v0Output = [FF32](repeating: 42, count: v0Net.destination.imageArea)
        var v1Output = [FF32](repeating: 42, count: v1Net.destination.imageArea)
        var positiveSlopeOutput = [FF32](repeating: 42, count: negativeSlopeNet.destination.imageArea)
        var negativeSlopeOutput = [FF32](repeating: 42, count: positiveSlopeNet.destination.imageArea)

        h0Net.activate(input: input, result: &h0Output)
        h1Net.activate(input: input, result: &h1Output)
        v0Net.activate(input: input, result: &v0Output)
        v1Net.activate(input: input, result: &v1Output)
        positiveSlopeNet.activate(input: input, result: &positiveSlopeOutput)
        negativeSlopeNet.activate(input: input, result: &negativeSlopeOutput)

        print(input)
        print(h0Output)
        print(h1Output)
        print(v0Output)
        print(v1Output)
        print(positiveSlopeOutput)
        print(negativeSlopeOutput)
    }
}

_ = Main()
