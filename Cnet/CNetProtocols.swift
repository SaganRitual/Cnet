// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

struct CNetIOSize {
    var area: Int { width * height }
    var volume: Int { area * channels }

    let channels: Int
    let height: Int
    let width: Int
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
