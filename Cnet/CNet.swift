// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CNetStructure {
    let descriptors: [CConvolution]
    let cKernelWeights: Int

    init(_ descriptors: [CConvolution]) {
        self.descriptors = descriptors
        self.cKernelWeights = descriptors.reduce(0) { $0 + $1.cKernelWeights }
    }
}

class CNet {
    let commandQueue: MTLCommandQueue

    let destination: CImage
    let source: CImage
    let netStructure: CNetStructure

    init(_ device: MTLDevice, structure: CNetStructure) {
        self.commandQueue = device.makeCommandQueue()!
        self.netStructure = structure

        self.source = structure.descriptors.first!.source
        self.destination = structure.descriptors.last!.destination
    }

    func activate(input: [FF32], result: inout [FF32]) {
        let commandBuffer = commandQueue.makeCommandBuffer()!

        source.inject(data: input)

        netStructure.descriptors[0].encode(
            to: commandBuffer,
            source: source,
            destination: destination
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        result.withUnsafeMutableBufferPointer { destination.extractData(to: $0) }
    }
}
