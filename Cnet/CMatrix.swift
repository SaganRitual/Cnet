// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CMatrix: CNetIO {
    let ioSize: CNetIOSize
    let device: MTLDevice

    let matrix: MPSMatrix

    init(
        _ device: MTLDevice, ioSize: CNetIOSize, data: [FF32]? = nil
    ) {
        self.ioSize = ioSize
        self.device = device

        let dd = MPSMatrixDescriptor(
            rows: ioSize.height, columns: ioSize.width,
            rowBytes: F16.bytesFF32(ioSize.width),
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
        assert(outputBuffer.count == ioSize.volume)

        let raw = UnsafeMutableRawPointer(outputBuffer.baseAddress!)
        raw.copyMemory(
            from: matrix.data.contents(),
            byteCount: F16.bytesFF32(outputBuffer.count)
        )
    }

    func inject(data: UnsafeBufferPointer<FF32>) {
        let ff16 =
            UnsafeMutableBufferPointer<FF16>.allocate(capacity: ioSize.volume)

        ff16.initialize(repeating: 0)

        F16.to16(from: data, result: ff16)

        matrix.data.contents().copyMemory(
            from: UnsafeRawPointer(data.baseAddress!),
            byteCount: F16.bytesFF32(data.count)
        )
    }

    func inject(data: [FF32]) {
        data.withUnsafeBufferPointer { inject(data: $0) }
    }
}
