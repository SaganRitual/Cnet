// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

struct CNetIOSize {
    var area: Int { width * height }
    var volume: Int { area * channels }

    var matrixColumns: Int { width }
    var matrixRows: Int { height }
    var vectorWidth: Int { width * height }

    let channels: Int
    let height: Int
    let width: Int

    init(channels: Int, height: Int, width: Int) {
        self.channels = channels
        self.height = height
        self.width = width
    }

    init(vectorWidth: Int) {
        self.channels = 1
        self.height = 1
        self.width = vectorWidth
    }

    init(matrixRows: Int, matrixColumns: Int) {
        self.channels = 1
        self.height = matrixRows
        self.width = matrixColumns
    }
}

protocol CNetIO {
    func extractData(to outputBuffer: UnsafeMutableBufferPointer<FF32>)

    func inject(data: [FF32])
    func inject(data: UnsafeBufferPointer<FF32>)
}

protocol CNetLayer {
    func encode(to commandBuffer: MTLCommandBuffer)
    func extractData(to outputBuffer: UnsafeMutableBufferPointer<FF32>)
    func getDestination() -> CNetIO
    func getSource() -> CNetIO
    func inject(data: UnsafeBufferPointer<FF32>)
}
