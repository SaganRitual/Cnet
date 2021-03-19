// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

print("Hello, World!")

class CNetStructure {
    let descriptors: [CConvolution]
    let cKernelWeights: Int

    init(_ descriptors: [CConvolution]) {
        self.descriptors = descriptors
        self.cKernelWeights = descriptors.reduce(0) { $0 + $1.cKernelWeights }
    }
}

enum Config {
    static let imageWidth = 4
    static let imageHeight = 4
    static let kernelWidth = 3
    static let kernelHeight = 3

    static var imageArea: Int { imageWidth * imageHeight }
    static var kernelArea: Int { kernelWidth * kernelHeight }
}

enum World {
    static let device = MTLCopyAllDevices()[0]
}

let kernelWeights =
    UnsafeMutableBufferPointer<FF32>.allocate(capacity: Config.kernelArea)

kernelWeights.initialize(repeating: 1)

let kernel = CConvolution(
    device: World.device, kernelWidth: Config.kernelWidth,
    kernelHeight: Config.kernelHeight,
    kernelWeights: UnsafeBufferPointer(kernelWeights)
)

let netStructure = CNetStructure([kernel])

let net = CNet(
    World.device,
    imageWidth: Config.imageWidth, imageHeight: Config.imageHeight,
    kernel: kernel
)

let input = [FF32](repeating: 1, count: Config.imageArea)
var output = [FF32](repeating: 42, count: Config.imageArea)
net.activate(input: input, result: &output)

print(output)
