// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CConvolution: NSObject {
    enum Tier { case top, hidden, bottom }

    var cKernelWeights: Int { kernelWidth * kernelHeight }

    private let kernelHeight: Int
    private let kernelWidth: Int
    private let imageWidth: Int
    private let imageHeight: Int

    var outputImageWidth: Int { 1 + imageWidth - kernelWidth }
    var outputImageHeight: Int { 1 + imageHeight - kernelHeight }

    private let kernel: MPSCNNConvolution
    private let dataSource: MPSCNNConvolutionDataSource

    let device: MTLDevice

    let destination: CImage
    let source: CImage
    let tier: Tier

    init(
        device: MTLDevice, tier: Tier,
        imageWidth: Int, imageHeight: Int,
        kernelWidth: Int, kernelHeight: Int,
        kernelWeights: UnsafeBufferPointer<FF32>
    ) {
        self.device = device
        self.tier = tier
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.kernelHeight = kernelHeight
        self.kernelWidth = kernelWidth

        self.source = CImage(device, imageWidth, imageHeight)

        let destinationWidth = 1 + imageWidth - kernelWidth
        let destinationHeight = 1 + imageHeight - kernelHeight
        self.destination = CImage(
            device, destinationWidth, destinationHeight
        )

        let d = CDatasource(kernelWidth, kernelHeight, kernelWeights)
        let c = MPSCNNConvolution(device: device, weights: d)

        c.clipRect = .init(
            origin: MTLOrigin(x: 0, y: 0, z: 0),

            size: MTLSize(
                width: destinationWidth, height: destinationHeight, depth: 1
            )
        )

        self.dataSource = d
        self.kernel = c

        c.offset.x = kernelWidth / 2
        c.offset.y = kernelHeight / 2

        super.init()
    }

    func encode(to cb: MTLCommandBuffer, source: CImage, destination: CImage) {
        kernel.encode(
            commandBuffer: cb, sourceImage: source.image,
            destinationImage: destination.image
        )
    }
}

class CDatasource: NSObject, MPSCNNConvolutionDataSource {
    private let convolutionDescriptor: MPSCNNConvolutionDescriptor!
    private let kernelWeights: UnsafeBufferPointer<FF32>

    init(
        _ kernelWidth: Int, _ kernelHeight: Int,
        _ kernelWeights: UnsafeBufferPointer<FF32>
    ) {
        self.kernelWeights = kernelWeights

        let d = MPSCNNConvolutionDescriptor(
            kernelWidth: kernelWidth, kernelHeight: kernelHeight,
            inputFeatureChannels: 1, outputFeatureChannels: 1, neuronFilter: nil
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
