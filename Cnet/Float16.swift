import Foundation
import Accelerate

typealias FF16 = UInt16
typealias FF32 = Float

enum F16 {
    static func to16(
        from input32: UnsafeBufferPointer<FF32>,
        result output16: UnsafeMutableBufferPointer<FF16>
    ) {
//        assert(input32.count == output16.count)
        var v32 = makev32Buffer(from: input32)
        var v16 = makev16Buffer(for: output16)
        vImageConvert_PlanarFtoPlanar16F(&v32, &v16, 0)
    }

    static func to32(
        from input16: UnsafeBufferPointer<FF16>,
        result output32: UnsafeMutableBufferPointer<FF32>
    ) {
        assert(input16.count == output32.count)
        var v16 = makev16Buffer(from: input16)
        var v32 = makev32Buffer(for: output32)
        vImageConvert_Planar16FtoPlanarF(&v16, &v32, 0)
    }
}

extension F16 {
    static func alignmentFF16() -> Int { MemoryLayout<FF16>.alignment }

    static func bytesFF16(_ elements: Int) -> vImagePixelCount {
        return vImagePixelCount(MemoryLayout<FF16>.size * elements)
    }

    static func bytesFF16(_ elements: Int) -> Int {
        MemoryLayout<FF16>.size * elements
    }

    static func alignmentFF32() -> Int { MemoryLayout<FF32>.alignment }

    static func bytesFF32(_ elements: Int) -> vImagePixelCount {
        vImagePixelCount(MemoryLayout<FF32>.size * elements)
    }

    static func bytesFF32(_ elements: Int) -> Int {
        MemoryLayout<FF32>.size * elements
    }
}

private extension F16 {

    typealias IB16 = UnsafeBufferPointer<FF16>
    typealias IB32 = UnsafeBufferPointer<FF32>

    typealias MB16 = UnsafeMutableBufferPointer<FF16>
    typealias MB32 = UnsafeMutableBufferPointer<FF32>

    typealias IR = UnsafeRawPointer // I for immutable
    typealias MR = UnsafeMutableRawPointer

    static func makev16Buffer(from input16: IB16) -> vImage_Buffer {
        vImage_Buffer(
            data: MR(mutating: input16.baseAddress!),
            height: 1, width: vImagePixelCount(input16.count),
            rowBytes: F16.bytesFF16(input16.count)
        )
    }

    static func makev16Buffer(for output16: MB16) -> vImage_Buffer {
        vImage_Buffer(
            data: MR(mutating: output16.baseAddress!),
            height: 1, width: vImagePixelCount(output16.count),
            rowBytes: F16.bytesFF16(output16.count)
        )
    }

    static func makev32Buffer(from input32: IB32) -> vImage_Buffer {
        vImage_Buffer(
            data: MR(mutating: input32.baseAddress!),
            height: 1, width: vImagePixelCount(input32.count),
            rowBytes: F16.bytesFF32(input32.count)
        )
    }

    static func makev32Buffer(for output32: MB32) -> vImage_Buffer {
        vImage_Buffer(
            data: MR(output32.baseAddress!),
            height: 1, width: vImagePixelCount(output32.count),
            rowBytes: F16.bytesFF32(output32.count)
        )
    }
}
