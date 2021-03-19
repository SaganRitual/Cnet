// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CImage {
    let imageHeight: Int
    let imageWidth: Int

    let device: MTLDevice
    let image: MPSImage
    let region: MTLRegion

    var imageArea: Int { imageHeight * imageWidth }

    init(_ device: MTLDevice, _ imageWidth: Int, _ imageHeight: Int) {
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight

        self.device = device

        let d = MPSImageDescriptor(
            channelFormat: .float16,
            width: imageWidth, height: imageHeight, featureChannels: 1
        )

        self.image = MPSImage(device: device, imageDescriptor: d)
        self.region = MTLRegionMake2D(0, 0, imageWidth, imageHeight)
    }

    func extractData(to outputBuffer: UnsafeMutableBufferPointer<FF32>) {
        assert(outputBuffer.count == imageArea)

        let bytesPerRow: Int = F16.bytesFF16(imageWidth)

        let ff16 = UnsafeMutableBufferPointer<FF16>.allocate(capacity: imageArea)

        image.texture.getBytes(
            UnsafeMutableRawPointer(ff16.baseAddress!),
            bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0
        )

        F16.to32(from: UnsafeBufferPointer(ff16), result: outputBuffer)

        ff16.deallocate()
    }

    func inject(data: [FF32]) {
        data.withUnsafeBufferPointer { input32 in
            let bytesPerRow: Int = F16.bytesFF16(imageWidth)

            let ff16 =
                UnsafeMutableBufferPointer<FF16>.allocate(capacity: imageArea)

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
