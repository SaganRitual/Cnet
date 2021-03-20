// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CFullyConnected: NSObject, CNetLayer {
    let destination: CMatrix
    let source: CMatrix

    let kernel: MPSMatrixMultiplication
    let weights: CMatrix

    func getDestination() -> CNetIO { destination }
    func getSource() -> CNetIO { source }

    init(
        _ device: MTLDevice, source: CNetIOSpec, destination: CNetIOSpec,
        weightsArray: [FF32]
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
    }

    func encode(to cb: MTLCommandBuffer) {
        kernel.encode(
            commandBuffer: cb,
            leftMatrix: source.matrix, rightMatrix: weights.matrix,
            resultMatrix: destination.matrix
        )
    }
}
