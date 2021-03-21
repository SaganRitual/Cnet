// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class CDatasource: NSObject, MPSCNNConvolutionDataSource {
    private let convolutionDescriptor: MPSCNNConvolutionDescriptor!
    private let kernelWeights: UnsafeBufferPointer<FF32>

    init(source: CNetIOSize, kernel: CNetIOSize, weights: UnsafeBufferPointer<FF32>) {
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
