// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CNet {
    let commandQueue: MTLCommandQueue

    var input: UnsafeMutableBufferPointer<FF32>
    var output: UnsafeMutableBufferPointer<FF32>

    let netStructure: [CNetLayer]

    init(
        _ device: MTLDevice, structure: [CNetLayer],
        input: UnsafeMutableBufferPointer<FF32>,
        output: UnsafeMutableBufferPointer<FF32>
    ) {
        self.commandQueue = device.makeCommandQueue()!
        self.input = input
        self.netStructure = structure
        self.output = output
    }

    func activate() {
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let inputLayer = netStructure.first!
        let outputLayer = netStructure.last!

        inputLayer.inject(data: UnsafeBufferPointer(input))
        inputLayer.encode(to: commandBuffer)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        outputLayer.extractData(to: output)
    }
}
