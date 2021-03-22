// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CNet {
    let commandQueue: MTLCommandQueue

    var input: UnsafeMutableBufferPointer<FF32>
    var output: UnsafeMutableBufferPointer<FF32>

    let convolution0: CWConvolution
    let convolution1: CWConvolution
    let fullyConnected: CFullyConnected

    init(
        _ device: MTLDevice,
        convolution0: CWConvolution, convolution1: CWConvolution,
        fullyConnected: CFullyConnected,
        input: UnsafeMutableBufferPointer<FF32>,
        output: UnsafeMutableBufferPointer<FF32>
    ) {
        self.commandQueue = device.makeCommandQueue()!
        self.input = input
        self.output = output
        self.convolution0 = convolution0
        self.convolution1 = convolution1
        self.fullyConnected = fullyConnected
    }

    func activate() {
        var commandBuffer = commandQueue.makeCommandBuffer()!

        convolution0.inject(data: UnsafeBufferPointer(rebasing: input[..<(input.count / 2)]))
        convolution1.inject(data: UnsafeBufferPointer(rebasing: input[(input.count / 2)...]))

        convolution0.encode(to: commandBuffer)
        convolution1.encode(to: commandBuffer)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let convolutionOutputs = UnsafeMutableBufferPointer<FF32>.allocate(
            capacity: fullyConnected.source.ioSize.volume
        )

        let t1 = UnsafeMutableBufferPointer(
            rebasing: convolutionOutputs[..<(convolutionOutputs.count / 2)]
        )

        let t2 = UnsafeMutableBufferPointer(
            rebasing: convolutionOutputs[(convolutionOutputs.count / 2)...]
        )

        convolution0.extractData(to: t1)
        convolution1.extractData(to: t2)

        fullyConnected.inject(data: UnsafeBufferPointer(convolutionOutputs))

        commandBuffer = commandQueue.makeCommandBuffer()!

        fullyConnected.encode(to: commandBuffer)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        fullyConnected.extractData(to: self.output)
    }

    func setStaticsBuffer(_ pStatics: UnsafeMutableBufferPointer<FF32>) {
        fullyConnected.setStaticsBuffer(pStatics)
    }
}
