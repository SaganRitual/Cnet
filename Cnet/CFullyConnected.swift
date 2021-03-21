// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CFullyConnected: NSObject, CNetLayer {
    let destination: CMatrix
    let source: CMatrix

    let kernel: MPSMatrixMultiplication
    let weights: CMatrix
    var biases: [FF32]

    var cBiases: Int { destination.ioSpec.width }
    var cWeights: Int { source.ioSpec.area * destination.ioSpec.width }

    func getDestination() -> CNetIO { destination }
    func getSource() -> CNetIO { source }

    init(
        _ device: MTLDevice, source: CNetIOSpec, destination: CNetIOSpec,
        weightsArray: [FF32], biasesArray: [FF32]? = nil
    ) {
        self.destination = CMatrix(device, ioSpec: destination)
        self.source = CMatrix(device, ioSpec: source)

        self.kernel = MPSMatrixMultiplication(
            device: device, transposeLeft: false, transposeRight: false,
            resultRows: destination.height, resultColumns: destination.width,
            interiorColumns: source.width, alpha: 1, beta: 1
        )

        let weightsSpec = CNetIOSpec(
            channels: 1, height: source.width, width: destination.width
        )

        self.weights = CMatrix(device, ioSpec: weightsSpec, data: weightsArray)
        self.biases = biasesArray ?? .init(repeating: 0, count: destination.width)
    }

    func encode(to cb: MTLCommandBuffer) {
        destination.inject(data: biases)

        kernel.encode(
            commandBuffer: cb,
            leftMatrix: source.matrix, rightMatrix: weights.matrix,
            resultMatrix: destination.matrix
        )
    }

    func setBiases(_ biases: UnsafeMutableBufferPointer<FF32>) {
        _ = self.biases.withUnsafeMutableBufferPointer {
            $0.initialize(from: biases)
        }
    }

    func setWeights(_ weights: UnsafeMutableBufferPointer<FF32>) {
        self.weights.matrix.data
            .contents()
            .assumingMemoryBound(to: FF32.self)
            .assign(from: UnsafePointer(weights.baseAddress!), count: weights.count)
    }
}
