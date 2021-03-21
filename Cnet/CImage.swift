// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CImage: CNetIO {
    let ioSize: CNetIOSize

    let device: MTLDevice
    let image: MPSImage
    let imageDescriptor: MPSImageDescriptor
    let region: MTLRegion

    init(_ device: MTLDevice, ioSize: CNetIOSize) {
        self.ioSize = ioSize
        self.device = device

        let d = MPSImageDescriptor(
            channelFormat: .float16,
            width: ioSize.width, height: ioSize.height,
            featureChannels: ioSize.channels
        )

        self.imageDescriptor = d
        self.region = MTLRegionMake2D(0, 0, ioSize.width, ioSize.height)
        self.image = MPSImage(device: device, imageDescriptor: d)
    }

    func extractData(to outputBuffer: UnsafeMutableBufferPointer<FF32>) {
        assert(outputBuffer.count == ioSize.volume)

        let bytesPerRow: Int = F16.bytesFF16(ioSize.width * ioSize.channels)

        let ff16 =
            UnsafeMutableBufferPointer<FF16>.allocate(capacity: ioSize.volume)

        image.texture.getBytes(
            UnsafeMutableRawPointer(ff16.baseAddress!),
            bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0
        )

        F16.to32(from: UnsafeBufferPointer(ff16), result: outputBuffer)

        ff16.deallocate()
    }

    func inject(data: UnsafeBufferPointer<FF32>) {
        let bytesPerRow: Int = F16.bytesFF16(ioSize.width * ioSize.channels)

        let ff16 =
            UnsafeMutableBufferPointer<FF16>.allocate(capacity: ioSize.volume)

        ff16.initialize(repeating: 0)

        F16.to16(from: data, result: ff16)

        let rr16 = UnsafeRawPointer(ff16.baseAddress!)

        image.texture.replace(
            region: region, mipmapLevel: 0,
            withBytes: rr16, bytesPerRow: bytesPerRow
        )

        ff16.deallocate()
    }

    func inject(data: [FF32]) {
        data.withUnsafeBufferPointer { inject(data: $0) }
    }
}
