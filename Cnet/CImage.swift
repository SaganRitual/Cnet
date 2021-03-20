// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

//protocol CImageProtocol {
//    var imageHeight: Int { get }
//    var imageWidth: Int { get }
//    var image: MPSImage? { get }
//    var imageDescriptor: MPSImageDescriptor { get }
//}
//
//class CTempImage: CImageProtocol {
//    let imageHeight: Int
//    let imageWidth: Int
//    var image: MPSImage?
//
//    let imageDescriptor: MPSImageDescriptor
//    let device: MTLDevice
//    let region: MTLRegion
//
//    init(_ device: MTLDevice, _ imageWidth: Int, _ imageHeight: Int) {
//        self.imageWidth = imageWidth
//        self.imageHeight = imageHeight
//
//        self.device = device
//
//        self.imageDescriptor = MPSImageDescriptor(
//            channelFormat: .float16,
//            width: imageWidth, height: imageHeight, featureChannels: 1
//        )
//
//        self.region = MTLRegionMake2D(0, 0, imageWidth, imageHeight)
//    }
//
//    func image(_ commandBuffer: MTLCommandBuffer) -> MPSTemporaryImage {
//        MPSTemporaryImage(
//            commandBuffer: commandBuffer, imageDescriptor: imageDescriptor
//        )
//    }
//}

class CImage {
    let ioSpec: CNetIO

    let device: MTLDevice
    let image: MPSImage
    let imageDescriptor: MPSImageDescriptor
    let region: MTLRegion

    init(_ device: MTLDevice, ioSpec: CNetIO) {
        self.ioSpec = ioSpec
        self.device = device

        let d = MPSImageDescriptor(
            channelFormat: .float16,
            width: ioSpec.width, height: ioSpec.height,
            featureChannels: ioSpec.channels
        )

        self.imageDescriptor = d
        self.image = MPSImage(device: device, imageDescriptor: d)
        self.region = MTLRegionMake2D(0, 0, ioSpec.width, ioSpec.height)
    }

    func extractData(to outputBuffer: UnsafeMutableBufferPointer<FF32>) {
        assert(outputBuffer.count == ioSpec.volume)

        let bytesPerRow: Int = F16.bytesFF16(ioSpec.width * ioSpec.channels)

        let ff16 =
            UnsafeMutableBufferPointer<FF16>.allocate(capacity: ioSpec.volume)

        image.texture.getBytes(
            UnsafeMutableRawPointer(ff16.baseAddress!),
            bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0
        )

        F16.to32(from: UnsafeBufferPointer(ff16), result: outputBuffer)

        ff16.deallocate()
    }

    func inject(data: [FF32]) {
        data.withUnsafeBufferPointer { input32 in
            let bytesPerRow: Int = F16.bytesFF16(ioSpec.width * ioSpec.channels)

            let ff16 =
                UnsafeMutableBufferPointer<FF16>.allocate(capacity: ioSpec.volume)

            ff16.initialize(repeating: 0)

            F16.to16(from: input32, result: ff16)

            let rr16 = UnsafeRawPointer(ff16.baseAddress!)

            image.texture.replace(
                region: region, mipmapLevel: 0,
                withBytes: rr16, bytesPerRow: bytesPerRow
            )

            ff16.deallocate()
        }
    }
}
