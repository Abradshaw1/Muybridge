//
//  ImageDisplayView.swift
//  ElifPose
//
//  Created by Elif Basokur on 17.07.2024.
//

import SwiftUI

struct ImageDisplayView: View {
    var image: UIImage
    var detectedKeypoints: [KeyPoint]
    var selectedSize: CGSize
    
    let viewSize = CGSize(width: 390, height: 520)
        
    var body: some View {
        VStack() {
            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .overlay(alignment: .topLeading){
                    GeometryReader{ geometry in
                        ZStack{
                            let path: Path = drawLines(detectedKeypoints, to: CGSize(width: geometry.size.width, height: geometry.size.height), originalSize: selectedSize)
                            path
                            .stroke(.gray,  lineWidth: 3)
                            
                            ForEach(scaleKeyPoints(detectedKeypoints, to: CGSize(width: geometry.size.width, height: geometry.size.height), originalSize: selectedSize), id: \.self) { keyPoint in
                                Circle()
                                    .fill(keyPoint.color)
                                    .frame(width: 10, height: 10)
                                    .position(keyPoint.cgPoint)
                             }
                            
                        }
                    }
                }
            Text(String(format: "MTU: %.2f", findMTU(detectedKeypoints, to:viewSize, originalSize: selectedSize)))
                .font(.system(size: 20, weight: .light, design: .serif))
                .italic()
                .ignoresSafeArea(.all)
        }
    }
    // Scaling key points method
    func scaleKeyPoints(_ keyPoints: [KeyPoint], to viewSize: CGSize, originalSize: CGSize) -> [KeyPoint] {
        return keyPoints.map {
            let scaleX = viewSize.width / originalSize.width
            let scaleY = viewSize.height / originalSize.height
            return KeyPoint($0.x * scaleX,  $0.y * scaleY, $0.color)
        }
    }
    func drawLines(_ keyPoints: [KeyPoint], to viewSize: CGSize, originalSize: CGSize) -> Path {
        var path = Path()
        let scaleX = viewSize.width / originalSize.width
        let scaleY = viewSize.height / originalSize.height
        
        if let firstPoint = detectedKeypoints.first {
            path.move(to: CGPoint(x: firstPoint.x * scaleX, y: firstPoint.y * scaleY))
        }
        path.addLine(to: CGPoint(x: detectedKeypoints[0].x * scaleX, y: detectedKeypoints[0].y * scaleY))
        path.addLine(to: CGPoint(x: detectedKeypoints[1].x * scaleX, y: detectedKeypoints[1].y * scaleY))

        path.addLine(to: CGPoint(x: detectedKeypoints[2].x * scaleX, y: detectedKeypoints[2].y * scaleY))
        path.addLine(to: CGPoint(x: detectedKeypoints[5].x * scaleX, y: detectedKeypoints[5].y * scaleY))
        path.addLine(to: CGPoint(x: detectedKeypoints[3].x * scaleX, y: detectedKeypoints[3].y * scaleY))
        path.move   (to: CGPoint(x: detectedKeypoints[5].x * scaleX, y: detectedKeypoints[5].y * scaleY))
        path.addLine(to: CGPoint(x: detectedKeypoints[4].x * scaleX, y: detectedKeypoints[4].y * scaleY))
        
        return path
    }
    
    func findMTU(_ keyPoints: [KeyPoint], to viewSize: CGSize, originalSize: CGSize) -> CGFloat {

        let scaleX = viewSize.width / originalSize.width
        let scaleY = viewSize.height / originalSize.height
        
        // MTU Estimation using Hawkings Hull formula

        let xDist: CGFloat    = (detectedKeypoints[1].x * scaleX - detectedKeypoints[2].x * scaleX); //[2]
        let yDist: CGFloat    = (detectedKeypoints[1].y * scaleY - detectedKeypoints[2].y * scaleY); //[3]
        let l_shank: CGFloat = sqrt((xDist * xDist) + (yDist * yDist));
        
        let alpha  = calculateAlpha(detectedKeypoints[0].cgPoint, detectedKeypoints[1].cgPoint, detectedKeypoints[2].cgPoint)

        let phi = calculatePhi(detectedKeypoints[1].cgPoint, detectedKeypoints[2].cgPoint, detectedKeypoints[5].cgPoint, detectedKeypoints[3].cgPoint)
        
        let MTU = l_shank * (0.9 + -6.2e-4 * alpha + 2.14e-3 * phi)
        return MTU
    }
    
    func calculateAlpha(_ pointA: CGPoint, _ pointB: CGPoint, _ pointC: CGPoint) -> CGFloat{
        
        let normA = CGPoint(x: pointA.x / selectedSize.width, y: pointA.y / selectedSize.height)
        let normB = CGPoint(x: pointB.x / selectedSize.width, y: pointB.y / selectedSize.height)
        let normC = CGPoint(x: pointC.x / selectedSize.width, y: pointC.y / selectedSize.height)
        
        let vec1 = CGPoint(x: normB.x - normA.x, y: normB.y - normA.y)
        let vec2 = CGPoint(x: normC.x - normB.x, y: normC.y - normB.y)
        
        let det = vec1.x * vec2.y - vec1.y * vec2.x
        let dot = vec1.x * vec2.x + vec1.y * vec2.y
        
        let alpha = abs(atan2(det, dot))
        
        return alpha
    }
    
    func calculatePhi(_ pointA: CGPoint, _ pointB: CGPoint, _ pointC: CGPoint, _ pointD: CGPoint) -> CGFloat{
    
        let normA = CGPoint(x: pointA.x / selectedSize.width, y: pointA.y / selectedSize.height)
        let normB = CGPoint(x: pointB.x / selectedSize.width, y: pointB.y / selectedSize.height)
        let normC = CGPoint(x: pointC.x / selectedSize.width, y: pointC.y / selectedSize.height)
        let normD = CGPoint(x: pointD.x / selectedSize.width, y: pointD.y / selectedSize.height)
        
        let vec1 = CGPoint(x: normB.x - normA.x, y: normB.y - normA.y)
        let vec2 = CGPoint(x: normC.x - normD.x, y: normC.y - normD.y)
        
        let det = vec1.x * vec2.y - vec1.y * vec2.x
        let dot = vec1.x * vec2.x + vec1.y * vec2.y

        let phi = abs(atan2(det, dot))
        
        return phi
}
    
    
}

struct ImageDisplayView_Previews: PreviewProvider {
    static var previews: some View {
        ImageDisplayView(image: UIImage(named: "example")!, detectedKeypoints: [], selectedSize: CGSize(width: 320, height: 320))
    }
}
