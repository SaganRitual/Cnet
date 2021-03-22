// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CWConvolution: NSObject, CNetLayer {
    enum Tier { case top, hidden, bottom }

    private let kernel: MPSCNNConvolution
    private let dataSource: MPSCNNConvolutionDataSource

    weak var device: MTLDevice!

    let destination: CImage
    let source: CImage

    func getDestination() -> CNetIO { destination }
    func getSource() -> CNetIO { source }

    init(
        _ device: MTLDevice,
        destinationIOSize: CNetIOSize, kernelIOSize: CNetIOSize, sourceIOSize: CNetIOSize,
        kernelWeights: UnsafeBufferPointer<FF32>
    ) {
        self.device = device

        self.source = CImage(device, ioSize: sourceIOSize)
        self.destination = CImage(device, ioSize: destinationIOSize)

        let d = CDatasource(
            source: sourceIOSize, kernel: kernelIOSize, weights: kernelWeights
        )

        let c = MPSCNNConvolution(device: device, weights: d)

        c.clipRect = .init(
            origin: MTLOrigin(x: 0, y: 0, z: 0),

            size: MTLSize(
                width: destinationIOSize.width,
                height: destinationIOSize.height, depth: 1
            )
        )

        self.dataSource = d
        self.kernel = c

        c.offset.x = -(destinationIOSize.width - 1) / 2
        c.offset.y = -(destinationIOSize.height - 1) / 2

        super.init()
    }

    func extractData(to outputBuffer: UnsafeMutableBufferPointer<FF32>) {
        destination.extractData(to: outputBuffer)
    }

    func inject(data: UnsafeBufferPointer<FF32>) {
        source.inject(data: data)
    }

    func encode(to cb: MTLCommandBuffer) {
        kernel.encode(
            commandBuffer: cb, sourceImage: source.image,
            destinationImage: destination.image
        )
    }
}
