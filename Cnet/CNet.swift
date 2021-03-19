// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CNet {
    let commandQueue: MTLCommandQueue

    let destination: CImage
    let kernel: CConvolution
    let source: CImage

    init(
        _ device: MTLDevice, imageWidth: Int, imageHeight: Int,
        kernel: CConvolution
    ) {
        self.commandQueue = device.makeCommandQueue()!

        self.source = CImage(device, imageWidth, imageHeight)
        self.destination = CImage(device, imageWidth, imageHeight)

        self.kernel = kernel
    }

    func activate(input: [FF32], result: inout [FF32]) {
        let commandBuffer = commandQueue.makeCommandBuffer()!

        source.inject(data: input)

        kernel.encode(
            to: commandBuffer, source: source, destination: destination
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        result.withUnsafeMutableBufferPointer { destination.extractData(to: $0) }
    }
}
