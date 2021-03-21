// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation

extension World {
    static let kernelWeights: [FF32] = [
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 8, 4, 2, 1,    // Horizontal

        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 8,    // Vertical
        0, 0, 0, 0, 0, 0, 4,    // e
        0, 0, 0, 0, 0, 0, 2,    // r
        0, 0, 0, 0, 0, 0, 1,    // t

        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 8, 0, 0, 0,    // Slope negative
        0, 0, 0, 0, 4, 0, 0,    //  l
        0, 0, 0, 0, 0, 2, 0,    //   o
        0, 0, 0, 0, 0, 0, 1,    //    p

        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 8,    // Slope positive
        0, 0, 0, 0, 0, 4, 0,    //   o
        0, 0, 0, 0, 2, 0, 0,    //  l
        0, 0, 0, 1, 0, 0, 0     // S
    ]
}