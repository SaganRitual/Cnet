// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

struct CNetIOSpec {
    var area: Int { width * height }
    var volume: Int { area * channels }

    let channels: Int
    let height: Int
    let width: Int
}

class CNet {
    let commandQueue: MTLCommandQueue

    let destination: CNetIO
    let source: CNetIO
    let netStructure: [CNetLayer]

    init(_ device: MTLDevice, structure: [CNetLayer]) {
        self.commandQueue = device.makeCommandQueue()!
        self.netStructure = structure

        self.source = structure.first!.getSource()
        self.destination = structure.last!.getDestination()
    }

    func activate(input: [FF32], result: inout [FF32]) {
        let commandBuffer = commandQueue.makeCommandBuffer()!

        source.inject(data: input)
        netStructure[0].encode(to: commandBuffer)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        result.withUnsafeMutableBufferPointer { destination.extractData(to: $0) }
    }
}
