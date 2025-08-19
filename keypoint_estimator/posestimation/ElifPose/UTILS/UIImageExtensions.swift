//
//  UIImageExtensions.swift
//  ElifPose
//
//  Created by Elif Basokur on 29.07.2024.
//

import UIKit
import CoreML

extension UIImage {
    /**
     Resizes the image.
     
     - Parameter scale: If this is 1, `newSize` is the size in pixels.
     */
    @nonobjc public func resized(to newSize: CGSize, scale: CGFloat = 1) -> UIImage {
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = scale
        let renderer = UIGraphicsImageRenderer(size: newSize, format: format)
        let image = renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: newSize))
        }
        return image
    }
    
    
    func mlMultiArray(scale preprocessScale:Float32=255, rBias preprocessRBias:Float32=0, gBias preprocessGBias:Float32=0, bBias preprocessBBias:Float32=0,
                      rScale preprocessRScale: Float32=1, gScale preprocessGScale: Float32=1, bScale preprocessBScale: Float32=1) -> MLMultiArray {
        let imagePixel = self.getPixelRgb(scale: preprocessScale, rBias: preprocessRBias, gBias: preprocessGBias, bBias: preprocessBBias,
                                          rScale: preprocessRScale, gScale: preprocessGScale, bScale: preprocessBScale)
        let size = self.size
        let imagePointer : UnsafePointer<Float32> = UnsafePointer(imagePixel)
        let mlArray = try! MLMultiArray(shape: [1, 3,  NSNumber(value: Float(size.height)), NSNumber(value: Float(size.width))], dataType: MLMultiArrayDataType.float32)
        mlArray.dataPointer.initializeMemory(as: Float32.self, from: imagePointer, count: imagePixel.count)
        return mlArray
    }
    
    func mlMultiArrayGrayScale(scale preprocessScale:Double=255,bias preprocessBias:Double=0) -> MLMultiArray {
        let imagePixel = self.getPixelGrayScale(scale: preprocessScale, bias: preprocessBias)
        let size = self.size
        let imagePointer : UnsafePointer<Double> = UnsafePointer(imagePixel)
        let mlArray = try! MLMultiArray(shape: [1, 1,  NSNumber(value: Float(size.width)), NSNumber(value: Float(size.height))], dataType: MLMultiArrayDataType.double)
        mlArray.dataPointer.initializeMemory(as: Double.self, from: imagePointer, count: imagePixel.count)
        return mlArray
    }

    func getPixelRgb(scale preprocessScale:Float32=255, rBias preprocessRBias:Float32=0, gBias preprocessGBias:Float32=0, bBias preprocessBBias:Float32=0,
                     rScale preprocessRScale: Float32=1, gScale preprocessGScale: Float32=1, bScale preprocessBScale: Float32=1) -> [Float32]
    {
        guard let cgImage = self.cgImage else {
            return []
        }
        let bytesPerRow = cgImage.bytesPerRow
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4

        let pixelData = cgImage.dataProvider!.data! as Data
        
        var r_buf : [Float32] = []
        var g_buf : [Float32] = []
        var b_buf : [Float32] = []
        
        for j in 0..<height {
            for i in 0..<width {
                let pixelInfo = bytesPerRow * j + i * bytesPerPixel

                let r = Float32(pixelData[pixelInfo])
                let g = Float32(pixelData[pixelInfo+1])
                let b = Float32(pixelData[pixelInfo+2])
                
                r_buf.append((Float32(r/preprocessScale)+preprocessRBias)/preprocessRScale)
                g_buf.append((Float32(g/preprocessScale)+preprocessGBias)/preprocessGScale)
                b_buf.append((Float32(b/preprocessScale)+preprocessBBias)/preprocessBScale)
            }
        }
        return ((b_buf + g_buf) + r_buf)
    }
    
    func getPixelGrayScale(scale preprocessScale:Double=255, bias preprocessBias:Double=0) -> [Double]
    {
        guard let cgImage = self.cgImage else {
            return []
        }
        let bytesPerRow = cgImage.bytesPerRow
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 2
        let pixelData = cgImage.dataProvider!.data! as Data
        
        var buf : [Double] = []
        
        for j in 0..<height {
            for i in 0..<width {
                let pixelInfo = bytesPerRow * j + i * bytesPerPixel
                let v = Double(pixelData[pixelInfo])
                buf.append(Double(v/preprocessScale)+preprocessBias)
            }
        }
        return buf
    }
    
}
