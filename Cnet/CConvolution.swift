// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CConvolution: NSObject {
    var cKernelWeights: Int { kernelWidth * kernelHeight }

    private let kernelHeight: Int
    private let kernelWidth: Int

    private let kernel: MPSCNNConvolution
    private let dataSource: MPSCNNConvolutionDataSource

    let device: MTLDevice

    init(
        device: MTLDevice,
        kernelWidth: Int, kernelHeight: Int,
        kernelWeights: UnsafeBufferPointer<FF32>
    ) {
        self.device = device
        self.kernelHeight = kernelHeight
        self.kernelWidth = kernelWidth

        let d = CDatasource(kernelWidth, kernelHeight, kernelWeights)
        let c = MPSCNNConvolution(device: device, weights: d)

        c.clipRect = .init(
            origin: MTLOrigin(x: 0, y: 0, z: 0),
            size: MTLSize(width: Config.imageWidth, height: Config.imageHeight, depth: 1)
        )

        self.dataSource = d
        self.kernel = c

        super.init()
    }

    func encode(
        to cb: MTLCommandBuffer, source: CImage, destination: CImage
    ) {
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
