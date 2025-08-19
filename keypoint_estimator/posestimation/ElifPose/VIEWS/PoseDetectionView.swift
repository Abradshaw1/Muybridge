//
//  PoseDetectionView.swift
//  ElifSemesterThesis
//
//  Created by Elif Basokur on 16.07.2024.
//

import SwiftUI
import Vision
import CoreML
import Accelerate
import CoreGraphics
import CoreVideo

struct PoseDetectionView: View {
    @Binding var useFrontCamera: Bool
    var DetModel: VNCoreMLModel
    var PoseModel: ElifPoseModel
    @State private var detectedKeypoints: [KeyPoint] = []
    @State private var frameSize: CGSize = .zero
    @State private var detectedBoxes: [CGRect] = []
    
    var body: some View {
        NavigationView { 
            VStack{
                ZStack{
                    CameraView(useFrontCamera: $useFrontCamera, onFrameCaptured: processFrame)
                        .edgesIgnoringSafeArea(.all)
                    
                    ForEach(scaleKeyPoints(detectedKeypoints, to: CGSize(width: 390, height: 844), originalSize: frameSize), id: \.self) { keyPoint in
                        Circle()
                            .fill(keyPoint.color)
                            .frame(width: 10, height: 10)
                            .position(keyPoint.cgPoint)
                            .edgesIgnoringSafeArea(.all)
                    }
                    let path: Path = drawLines(detectedKeypoints, to: CGSize(width: 390, height: 844), originalSize: frameSize)
                    path
                    .stroke(.gray,  lineWidth: 3)
                }
                .navigationBarTitle("Pose Detection", displayMode: .inline)  // Add navigation bar title
                Text(String(format: "MTU: %.2f", findMTU(detectedKeypoints, to: CGSize(width: 390, height: 844), originalSize: frameSize)))
                    .font(.system(size: 20, weight: .light, design: .serif))
                    .italic()
                    .ignoresSafeArea(.all)
            }
        }
    }

    func processFrame(pixelBuffer: CVPixelBuffer) {
        // Get size of the image
        guard let uiImage = UIImage(pixelBuffer: pixelBuffer) else { print("Could not capture image"); return }
        detectedBoxes = []
        
        let imageWidth  = uiImage.size.height//Double(CVPixelBufferGetWidth(pixelBuffer))
        let imageHeight = uiImage.size.width//Double(CVPixelBufferGetHeight(pixelBuffer))
        
        frameSize = CGSize(width: imageWidth, height: imageHeight)
        
        var rotationAngle = vImage.Rotation.clockwise90Degrees
        var rgbDestinationImageFormat = vImage_CGImageFormat(
            bitsPerComponent: 8,
            bitsPerPixel: 8 * 4,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)
        )!
        let pixelFormat = vImage.Interleaved8x4.self

        let rgbDestinationBuffer = try! vImage.PixelBuffer(
            cgImage: uiImage.cgImage!,
            cgImageFormat: &rgbDestinationImageFormat,
            pixelFormat: pixelFormat)
        
        
        let resizedBuffer = vImage.PixelBuffer<vImage.Interleaved8x4>(size: vImage.Size(width: 640, height: 640))
        // Scale image
        rgbDestinationBuffer.scale(destination: resizedBuffer)
        
        
        // Rotate RGBDestinationBuffer 90 degrees to the right since for some reason UIImages are stored 90 degrees pre-rotated
        let correctedBuffer = vImage.PixelBuffer<vImage.Interleaved8x4>(
            size: vImage.Size(width: 640, height: 640))
        
        resizedBuffer.rotate(rotationAngle, destination: correctedBuffer)
        let recognitions = VNCoreMLRequest(model: DetModel, completionHandler: detectionDidComplete)
        let requests = [recognitions]
        
        correctedBuffer.withCVPixelBuffer(readOnly: false) { cvPixelBuffer in
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: cvPixelBuffer, orientation: .up, options: [:])
            try! imageRequestHandler.perform(requests)
        }
        
        let count = detectedBoxes.count
        
        // Extract bounding box from results
        detectedKeypoints = []
        for i in 0..<count {
            
            let padding: CGFloat = 10
            // Extract detection bounding box, denormalize and add padding
            let bbox_x  = max(detectedBoxes[i].minX * imageWidth - padding, 0)
            let bbox_y  = imageHeight - max(detectedBoxes[i].minY * imageHeight - padding, 0)
            let bbox_x2 = min(detectedBoxes[i].maxX * imageWidth + padding,  imageWidth)
            let bbox_y2 = imageHeight - min(detectedBoxes[i].maxY * imageHeight + padding, imageHeight)
            
            let bbox_w = bbox_x2 - bbox_x
            let bbox_h = bbox_y  - bbox_y2
            
            // Do pose estimation if the bbox is large enough
            let bbox_area = detectedBoxes[i].width * detectedBoxes[i].height
            
            if bbox_area > 0.1 && i == 0{
                
                // Crop the detected person
                // Define region of interest
                let roi_x  = Int(floor(max(detectedBoxes[i].minX * 640 - padding, 0)))
                let roi_y  = Int(floor(640 - max(detectedBoxes[i].minY * 640 - padding, 0)))
                let roi_x2 = Int(floor(min(detectedBoxes[i].maxX * 640 + padding,  640)))
                let roi_y2 = Int(floor(640 - min(detectedBoxes[i].maxY * 640 + padding, 640)))
                
                let roi_w = roi_x2 - roi_x
                let roi_h = roi_y  - roi_y2
                
                let roi = CGRect(x: roi_x, y: roi_y2, width: roi_w, height: roi_h)
                                
                let croppedImageBuffer = correctedBuffer.cropped(to: roi)
                // Scale cropped image to the size of the input format for Pose Esimation ML model
                // ElifPose model expects images of size (192, 256)
                
                // Create memory for storing rescaled image
                let resizedPoseBuffer = vImage.PixelBuffer<vImage.Interleaved8x4>(size: vImage.Size(width: 192, height: 256))
                
                // Scale image
                croppedImageBuffer.scale(destination: resizedPoseBuffer)
                let resizedPoseImage = resizedPoseBuffer.makeCGImage(cgImageFormat: rgbDestinationImageFormat)
                let RTMPose_in = UIImage(cgImage: resizedPoseImage!).mlMultiArray(scale: 1, rBias: -123.675, gBias: -116.28, bBias: -103.53, rScale: 58.395, gScale: 57.12, bScale: 57.375)
                let pose_input = ElifPoseModelInput(input: RTMPose_in)
                let pose_result = try! PoseModel.prediction(input: pose_input)
                
                let simcc_x = pose_result.simcc_x
                let simcc_y = pose_result.simcc_y
                
                let no_keypoints = 26
                let left_leg_keypoints = [11, 13, 15, 20, 22, 24]
                let right_leg_keypoints = [12, 14, 16, 21, 23, 25]
                let leg_colors = [Color.red, Color.blue, Color.pink, Color.orange, Color.green, Color.yellow]
                var color_idx = 0
                
                for i in 0..<no_keypoints {
                    
                    if left_leg_keypoints.contains(i){
                        
                        let x_ptr = UnsafePointer<Float>(OpaquePointer(simcc_x.dataPointer))
                        let x_info = argmax(x_ptr.advanced(by: i * 192 * 2), count: 192*2, stride: 1)
                        let y_ptr = UnsafePointer<Float>(OpaquePointer(simcc_y.dataPointer))
                        let y_info = argmax(y_ptr.advanced(by: i * 256 * 2), count: 256*2, stride: 1)
                        let x_coord = Double(x_info.0) / 2
                        let y_coord = Double(y_info.0) / 2
                        
                        let roi_wd = Double(roi_w)
                        let roi_hd = Double(roi_h)
                        let roi_xd = Double(roi_x)
                        let roi_y2d = Double(roi_y2)
                        
                        let bbox_xcoord = min(Double(x_coord * roi_wd) / 192.0, roi_wd)  +  roi_xd
                        let bbox_ycoord = min(Double(y_coord * roi_hd) / 256.0, roi_hd)  +  roi_y2d
                        
                        let resized_xcoord = min(bbox_xcoord / 640 * imageWidth, imageWidth)
                        let resized_ycoord = min(bbox_ycoord / 640 * imageHeight, imageHeight)
                        
                        let keypoint_color = leg_colors[color_idx]
                        color_idx = color_idx + 1
                        
                        let keypoint = KeyPoint(resized_xcoord, resized_ycoord, keypoint_color)
                        detectedKeypoints.append(keypoint)
                    }
                }
            }
        }
    }
        
        
        func cropImage(image: UIImage, rect: CGRect, scale: CGFloat) -> UIImage? {
            UIGraphicsBeginImageContextWithOptions(CGSize(width: rect.size.width / scale, height: rect.size.height / scale), true, 0.0)
            image.draw(at: CGPoint(x: -rect.origin.x / scale, y: -rect.origin.y / scale))
            let croppedImage = UIGraphicsGetImageFromCurrentImageContext()
            UIGraphicsEndImageContext()
            return croppedImage
        }
        
        func detectionDidComplete(request: VNRequest, error: Error?) {
            if let results = request.results {
                for observation in results where observation is VNRecognizedObjectObservation {
                    guard let objectObservation = observation as? VNRecognizedObjectObservation else { continue }
                    if objectObservation.labels[0].identifier == "person" && objectObservation.labels[0].confidence > 0.6 {
                        detectedBoxes.append(objectObservation.boundingBox)
                    }
                }
            }
        }
        // Scaling key points method
        func scaleKeyPoints(_ keyPoints: [KeyPoint], to viewSize: CGSize, originalSize: CGSize) -> [KeyPoint] {
            return keyPoints.map {
                let scaleX = viewSize.width / originalSize.width
                let scaleY = viewSize.height / originalSize.height
                return KeyPoint(($0.x-46) * scaleX,  $0.y * scaleY, $0.color)
            }
        }
    
        func drawLines(_ keyPoints: [KeyPoint], to viewSize: CGSize, originalSize: CGSize) -> Path {
            var path = Path()
            
            if keyPoints.count == 6 {
                
                let scaleX = viewSize.width / originalSize.width
                let scaleY = viewSize.height / originalSize.height
                
                if let firstPoint = detectedKeypoints.first {
                    path.move(to: CGPoint(x: (firstPoint.x-46) * scaleX, y: firstPoint.y * scaleY-133))
                }
                path.addLine(to: CGPoint(x: (detectedKeypoints[0].x-46) * scaleX, y: detectedKeypoints[0].y * scaleY-133))
                path.addLine(to: CGPoint(x: (detectedKeypoints[1].x-46) * scaleX, y: detectedKeypoints[1].y * scaleY-133))
                path.addLine(to: CGPoint(x: (detectedKeypoints[2].x-46) * scaleX, y: detectedKeypoints[2].y * scaleY-133))
                path.addLine(to: CGPoint(x: (detectedKeypoints[5].x-46) * scaleX, y: detectedKeypoints[5].y * scaleY-133))
                path.addLine(to: CGPoint(x: (detectedKeypoints[3].x-46) * scaleX, y: detectedKeypoints[3].y * scaleY-133))
                path.move   (to: CGPoint(x: (detectedKeypoints[5].x-46) * scaleX, y: detectedKeypoints[5].y * scaleY-133))
                path.addLine(to: CGPoint(x: (detectedKeypoints[4].x-46) * scaleX, y: detectedKeypoints[4].y * scaleY-133))
            }
            
            return path
        }
    
        func findMTU(_ keyPoints: [KeyPoint], to viewSize: CGSize, originalSize: CGSize) -> CGFloat {
            
            if keyPoints.count == 6 {
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
            else {
                return 0
            }
            
        }
    
        func calculateAlpha(_ pointA: CGPoint, _ pointB: CGPoint, _ pointC: CGPoint) -> CGFloat{
            
            let normA = CGPoint(x: pointA.x / frameSize.width, y: pointA.y / frameSize.height)
            let normB = CGPoint(x: pointB.x / frameSize.width, y: pointB.y / frameSize.height)
            let normC = CGPoint(x: pointC.x / frameSize.width, y: pointC.y / frameSize.height)
            
            let vec1 = CGPoint(x: normB.x - normA.x, y: normB.y - normA.y)
            let vec2 = CGPoint(x: normC.x - normB.x, y: normC.y - normB.y)
            
            let det = vec1.x * vec2.y - vec1.y * vec2.x
            let dot = vec1.x * vec2.x + vec1.y * vec2.y
            
            let alpha = abs(atan2(det, dot))
            
            return alpha
        }
        
        func calculatePhi(_ pointA: CGPoint, _ pointB: CGPoint, _ pointC: CGPoint, _ pointD: CGPoint) -> CGFloat{
        
            let normA = CGPoint(x: pointA.x / frameSize.width, y: pointA.y / frameSize.height)
            let normB = CGPoint(x: pointB.x / frameSize.width, y: pointB.y / frameSize.height)
            let normC = CGPoint(x: pointC.x / frameSize.width, y: pointC.y / frameSize.height)
            let normD = CGPoint(x: pointD.x / frameSize.width, y: pointD.y / frameSize.height)
            
            let vec1 = CGPoint(x: normB.x - normA.x, y: normB.y - normA.y)
            let vec2 = CGPoint(x: normC.x - normD.x, y: normC.y - normD.y)
            
            let det = vec1.x * vec2.y - vec1.y * vec2.x
            let dot = vec1.x * vec2.x + vec1.y * vec2.y

            let phi = abs(atan2(det, dot))
            
            return phi
        }
    
}
    
    


struct PoseDetectionView_Previews: PreviewProvider {
    @State static var useFrontCamera = true

    static var previews: some View {
        PoseDetectionView(
            useFrontCamera: $useFrontCamera,
            DetModel: try! VNCoreMLModel(for: MLModel(contentsOf: Bundle.main.url(forResource: "yolov8n", withExtension: "mlmodelc")!)),
            PoseModel: try! ElifPoseModel()
        )
    }
}




