// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CFullyConnected: NSObject, CNetLayer {
    let destination: CMatrix
    let source: CMatrix

    let kernel: MPSMatrixMultiplication
    let weights: CMatrix
    var pBiases: UnsafeBufferPointer<FF32>!

    var cBiases: Int { destination.ioSize.width }
    var cWeights: Int { source.ioSize.area * destination.ioSize.width }

    func getDestination() -> CNetIO { destination }
    func getSource() -> CNetIO { source }

    init(
        _ device: MTLDevice,
        destinationIOSize: CNetIOSize,
        sourceIOSize: CNetIOSize
    ) {
        self.destination = CMatrix(device, ioSize: destinationIOSize)
        self.source = CMatrix(device, ioSize: sourceIOSize)

        self.kernel = MPSMatrixMultiplication(
            device: device, transposeLeft: false, transposeRight: false,
            resultRows: destinationIOSize.matrixRows,
            resultColumns: destinationIOSize.matrixColumns,
            interiorColumns: sourceIOSize.vectorWidth, alpha: 1, beta: 1
        )

        let weightsSpec = CNetIOSize(
            channels: 1,
            height: sourceIOSize.matrixColumns,
            width: destinationIOSize.matrixColumns
        )

        self.weights = CMatrix(device, ioSize: weightsSpec)
    }

    func encode(to cb: MTLCommandBuffer) {
        destination.inject(data: pBiases)

        kernel.encode(
            commandBuffer: cb,
            leftMatrix: source.matrix, rightMatrix: weights.matrix,
            resultMatrix: destination.matrix
        )
    }

    func extractData(to outputBuffer: UnsafeMutableBufferPointer<FF32>) {
        destination.extractData(to: outputBuffer)
    }

    func inject(data: UnsafeBufferPointer<FF32>) { source.inject(data: data) }

    func setStaticsBuffer(_ pStatics: UnsafeMutableBufferPointer<FF32>) {
        self.pBiases = UnsafeBufferPointer<FF32>(rebasing: pStatics[0..<cBiases])
        setWeights(UnsafeBufferPointer<FF32>(rebasing: pStatics[cBiases...]))
    }

    func setWeights(_ weights: UnsafeBufferPointer<FF32>) {
        self.weights.matrix.data
            .contents()
            .assumingMemoryBound(to: FF32.self)
            .assign(from: UnsafePointer(weights.baseAddress!), count: weights.count)
    }
}
