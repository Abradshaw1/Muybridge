//
//  Keypoint.swift
//  ElifPose
//
//  Created by Elif Basokur on 19.07.2024.
//

import Foundation
import CoreGraphics
import SwiftUI

struct KeyPoint: Hashable {
    let x: Double
    let y: Double
    let color: Color

    init(_ x: Double, _ y: Double, _ color: Color) {
        self.x = x
        self.y = y
        self.color = color
    }

    init(_ point: CGPoint, _ color: Color) {
        self.x = Double(point.x)
        self.y = Double(point.y)
        self.color = color
    }

    var cgPoint: CGPoint {
        return CGPoint(x: x, y: y)
    }
    
    
}
