// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CMatrix: CNetIO {
    let ioSpec: CNetIOSpec
    let device: MTLDevice

    let matrix: MPSMatrix

    init(
        _ device: MTLDevice, ioSpec: CNetIOSpec, data: [FF32]? = nil
    ) {
        self.ioSpec = ioSpec
        self.device = device

        let dd = MPSMatrixDescriptor(
            rows: ioSpec.height, columns: ioSpec.width,
            rowBytes: F16.bytesFF32(ioSpec.width),
            dataType: .float32
        )

        guard let data = data else {
            self.matrix = MPSMatrix(device: device, descriptor: dd)
            return
        }

        let bb = device.makeBuffer(
            length: F16.bytesFF32(data.count),
            options: .storageModeShared
        )!

        data.withUnsafeBytes {
            bb.contents().copyMemory(
                from: $0.baseAddress!,
                byteCount: F16.bytesFF32(data.count)
            )
        }

        self.matrix = MPSMatrix(buffer: bb, descriptor: dd)
    }
}

extension CMatrix {
    func extractData(to outputBuffer: UnsafeMutableBufferPointer<FF32>) {
        assert(outputBuffer.count == ioSpec.volume)

//        let bytesPerRow: Int = F16.bytesFF16(ioSpec.width * ioSpec.channels)

//        let ff16 =
//            UnsafeMutableBufferPointer<FF16>.allocate(capacity: ioSpec.volume)

        let raw = UnsafeMutableRawPointer(outputBuffer.baseAddress!)
        raw.copyMemory(
            from: matrix.data.contents(),
            byteCount: F16.bytesFF32(outputBuffer.count)
        )

//        F16.to32(from: UnsafeBufferPointer(ff16), result: outputBuffer)

//        ff16.deallocate()
    }

    func inject(data: [FF32]) {
        data.withUnsafeBufferPointer { input32 in
//            let bytesPerRow: Int = F16.bytesFF16(ioSpec.width * ioSpec.channels)

            let ff16 =
                UnsafeMutableBufferPointer<FF16>.allocate(capacity: ioSpec.volume)

            ff16.initialize(repeating: 0)

            F16.to16(from: input32, result: ff16)

//            let rr16 = UnsafeRawPointer(ff16.baseAddress!)

            data.withUnsafeBytes {
                matrix.data.contents().copyMemory(
                    from: $0.baseAddress!,
                    byteCount: F16.bytesFF32(data.count)
                )
            }

//            ff16.deallocate()
        }
    }
}
