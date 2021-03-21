// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CNet {
    let commandQueue: MTLCommandQueue

    var input: UnsafeMutableBufferPointer<FF32>
    var output: UnsafeMutableBufferPointer<FF32>

    let convolution0: CWConvolution
    let convolution1: CWConvolution

    init(
        _ device: MTLDevice,
        convolution0: CWConvolution, convolution1: CWConvolution,
        input: UnsafeMutableBufferPointer<FF32>,
        output: UnsafeMutableBufferPointer<FF32>
    ) {
        self.commandQueue = device.makeCommandQueue()!
        self.input = input
        self.output = output
        self.convolution0 = convolution0
        self.convolution1 = convolution1
    }

    func activate() {
        let commandBuffer = commandQueue.makeCommandBuffer()!

        convolution0.inject(data: UnsafeBufferPointer(rebasing: input[..<(input.count / 2)]))
        convolution1.inject(data: UnsafeBufferPointer(rebasing: input[(input.count / 2)...]))

        convolution0.encode(to: commandBuffer)
        convolution1.encode(to: commandBuffer)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        convolution0.extractData(to: UnsafeMutableBufferPointer(rebasing: output[..<(output.count / 2)]))
        convolution1.extractData(to: UnsafeMutableBufferPointer(rebasing: output[(output.count / 2)...]))
    }
}
