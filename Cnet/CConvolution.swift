// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

protocol CNetLayer {
    func encode(to commandBuffer: MTLCommandBuffer)
    func getDestination() -> CNetIO
    func getSource() -> CNetIO
}

class CConvolution: NSObject, CNetLayer {
    enum Tier { case top, hidden, bottom }

    private let kernel: MPSCNNConvolution
    private let dataSource: MPSCNNConvolutionDataSource

    let cKernelWeights: Int

    weak var device: MTLDevice!

    let destination: CImage
    let source: CImage
    let tier: Tier

    func getDestination() -> CNetIO { destination }
    func getSource() -> CNetIO { source }

    init(
        device: MTLDevice, tier: Tier,
        destinationIoSpec: CNetIOSpec, kernelIoSpec: CNetIOSpec, sourceIoSpec: CNetIOSpec,
        kernelWeights: UnsafeBufferPointer<FF32>
    ) {
        self.device = device
        self.tier = tier
        self.cKernelWeights = kernelIoSpec.volume

        self.source = CImage(device, ioSpec: sourceIoSpec)
        self.destination = CImage(device, ioSpec: destinationIoSpec)

        let d = CDatasource(
            source: sourceIoSpec, kernel: kernelIoSpec, weights: kernelWeights
        )

        let c = MPSCNNConvolution(device: device, weights: d)

        c.clipRect = .init(
            origin: MTLOrigin(x: 0, y: 0, z: 0),

            size: MTLSize(
                width: destinationIoSpec.width,
                height: destinationIoSpec.height, depth: 1
            )
        )

        self.dataSource = d
        self.kernel = c

        c.offset.x = -(destinationIoSpec.width - 1) / 2
        c.offset.y = -(destinationIoSpec.height - 1) / 2

        super.init()
    }

    func encode(to cb: MTLCommandBuffer) {
        kernel.encode(
            commandBuffer: cb, sourceImage: source.image,
            destinationImage: destination.image
        )
    }
}

class CDatasource: NSObject, MPSCNNConvolutionDataSource {
    private let convolutionDescriptor: MPSCNNConvolutionDescriptor!
    private let kernelWeights: UnsafeBufferPointer<FF32>

    init(source: CNetIOSpec, kernel: CNetIOSpec, weights: UnsafeBufferPointer<FF32>) {
        self.kernelWeights = weights

        let d = MPSCNNConvolutionDescriptor(
            kernelWidth: kernel.width, kernelHeight: kernel.height,
            inputFeatureChannels: source.channels,
            outputFeatureChannels: kernel.channels,
            neuronFilter: nil
        )

        d.strideInPixelsX = 1
        d.strideInPixelsY = 1

        self.convolutionDescriptor = d

        super.init()
    }

    func biasTerms() -> UnsafeMutablePointer<Float>? { nil }
    func dataType() -> MPSDataType { .float32 }
    func descriptor() -> MPSCNNConvolutionDescriptor { convolutionDescriptor }
    func load() -> Bool { true }
    func purge() { }
    func label() -> String? { nil }
    func copy(with zone: NSZone? = nil) -> Any { false }

    func weights() -> UnsafeMutableRawPointer {
        UnsafeMutableRawPointer(mutating: kernelWeights.baseAddress!)
    }
}
