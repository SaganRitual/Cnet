// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

print("Hello, World!")

enum Config {
    static let winLength = 4
    static let imageWidth = 7
    static let imageHeight = 6
    static let kernelWidth = winLength
    static let kernelHeight = winLength

    static var imageArea: Int { imageWidth * imageHeight }
    static var kernelArea: Int { kernelWidth * kernelHeight }
}

enum World {
    static let device = MTLCopyAllDevices()[0]
}

let horizontalWeights =
    UnsafeMutableBufferPointer<FF32>.allocate(capacity: Config.winLength)

horizontalWeights.initialize(repeating: 1)

let verticalWeights =
    UnsafeMutableBufferPointer<FF32>.allocate(capacity: Config.winLength)

verticalWeights.initialize(repeating: 1)

let negativeSlopeWeights = UnsafeMutableBufferPointer<FF32>.allocate(
    capacity: Config.kernelArea
)

_ = negativeSlopeWeights.initialize(from: [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
])

let positiveSlopeWeights = UnsafeMutableBufferPointer<FF32>.allocate(
    capacity: Config.kernelArea
)

_ = positiveSlopeWeights.initialize(from: [
    0, 0, 0, 1,
    0, 0, 1, 0,
    0, 1, 0, 0,
    1, 0, 0, 0
])

//let horizontalKernel = CConvolution(
//    device: World.device, tier: .top,
//    imageWidth: Config.imageWidth, imageHeight: Config.imageHeight,
//    kernelWidth: Config.kernelWidth, kernelHeight: Config.kernelHeight,
//    kernelWeights: UnsafeBufferPointer(horizontalWeights)
//)
//
//let verticalKernel = CConvolution(
//    device: World.device, tier: .top,
//    imageWidth: Config.imageWidth, imageHeight: Config.imageHeight,
//    kernelWidth: Config.kernelWidth, kernelHeight: Config.kernelHeight,
//    kernelWeights: UnsafeBufferPointer(verticalWeights)
//)
//
//let negativeSlopeKernel = CConvolution(
//    device: World.device, tier: .top,
//    imageWidth: Config.imageWidth, imageHeight: Config.imageHeight,
//    kernelWidth: Config.kernelWidth, kernelHeight: Config.kernelHeight,
//    kernelWeights: UnsafeBufferPointer(negativeSlopeWeights)
//)

let positiveSlopeKernel = CConvolution(
    device: World.device, tier: .top,
    imageWidth: Config.imageWidth, imageHeight: Config.imageHeight,
    kernelWidth: Config.kernelWidth, kernelHeight: Config.kernelHeight,
    kernelWeights: UnsafeBufferPointer(positiveSlopeWeights)
)

let netStructure = CNetStructure([
    /*horizontalKernel, verticalKernel, negativeSlopeKernel, */positiveSlopeKernel
])

let net = CNet(World.device, structure: netStructure)

let input: [FF32] = (0..<Config.imageArea).map { _ in FF32(Int.random(in: -1...1)) }
var output = [FF32](repeating: 42, count: 12)
net.activate(input: input, result: &output)

print(input)
print(output)
