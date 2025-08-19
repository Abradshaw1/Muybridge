//
//  HomeView.swift
//  ElifSemesterThesis
//
//  Created by Elif Basokur on 16.07.2024.
//

// Home view, that will be shown when app is launched

import SwiftUI
import PhotosUI
import Vision
import CoreML
import Accelerate
import CoreGraphics
import CoreVideo

struct HomeView: View {
    @State private var showSettings = false
    @State private var showGuidance = false
    @State private var useFrontCamera = false
    @State private var showImagePicker = false
    @State private var selectedImage: UIImage?
    @State private var shownImage: UIImage?
    
    @State private var showImageDisplay = false
    @State private var detectedKeypoints: [KeyPoint] = []
    @State private var detectedBoxes: [CGRect] = []
    @State private var croppedBbox: CGRect = .zero
    
    @State private var selectedSize: CGSize = .zero
    @State private var DetModel = try! VNCoreMLModel(for: MLModel(contentsOf: Bundle.main.url(forResource: "ElifDetModel", withExtension: "mlmodelc")!))
    @State private var PoseModel: ElifPoseModel = try! ElifPoseModel()
    @State private var isPickerPresented = false
    @State private var selectedVideoURL: URL?
    @State private var showVideoDisplay = false
    @State private var KeyPointHistory: [[KeyPoint]] = [[]]
    @State private var smoothKeyPoints: [[KeyPoint]] = [[]]
    @State private var videoFrameRate: Double = 30.0
    
    @State private var processingProgress: Double = 0.0
    @State private var isProcessing: Bool = false
    
    var body: some View {
        NavigationView {
            VStack {
                HStack {
                    Button(action: {
                        showSettings.toggle()
                    }) {
                        Image(systemName: "gear")
                            .font(.system(size: 40))
                            .padding()
                            .background(Color(.purple))
                            .clipShape(Circle())
                            .foregroundColor(.white)
                    }
                    Spacer()
                    Button(action: {
                        showGuidance.toggle()
                    }) {
                        Image(systemName: "book")
                            .font(.system(size: 40))
                            .padding()
                            .background(Color(.purple))
                            .clipShape(Circle())
                            .foregroundColor(.white)
                    }
                }
                .padding()
                
                Spacer()
                
                Image("AppLogoMiddle")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 300, height: 300)
                    .clipShape(Circle())
                    .overlay(Circle().stroke(Color.white, lineWidth: 4))
                    .shadow(radius: 10)
                    .padding()
                
                Spacer()
                
                NavigationLink(destination: PoseDetectionView(
                    useFrontCamera: $useFrontCamera,
                    DetModel: DetModel,
                    PoseModel: PoseModel
                )) {
                    HStack {
                        Text("Detect Pose")
                        Image(systemName: "camera")
                    }
                    .font(.title)
                    .foregroundColor(.white)
                    .padding()
                    .background(Color(.purple))
                    .cornerRadius(20)
                }
                .padding(.bottom)
                
                HStack{
                    Button(action: {
                        showImagePicker = true
                    }) {
                        HStack {
                            Text("Upload")
                            Image(systemName: "photo")
                        }
                        .font(.title)
                        .foregroundColor(.white)
                        .padding()
                        .background(Color(.purple))
                        .cornerRadius(20)
                    }
                    .sheet(isPresented: $showImagePicker) {
                        ImagePicker(selectedImage: $selectedImage)
                    }
                    .onChange(of: selectedImage) { newValue in
                        if let image = newValue {
                            print("Changed Image")
                            processImage(image)
                            showImageDisplay = true
                        }
                        else{
                            print("Not changed")
                        }
                    }
                    .background(
                        NavigationLink(destination: ImageDisplayView(image: selectedImage ?? UIImage(), detectedKeypoints: detectedKeypoints, selectedSize: selectedSize), isActive: $showImageDisplay) {
                            EmptyView()
                        }
                    )
                    Button(action: {
                        isPickerPresented = true
                        isProcessing = true
                        
                    }) {
                        HStack {
                            Text("Video")
                            Image(systemName: "video")
                        }
                        .font(.title)
                        .foregroundColor(.white)
                        .padding()
                        .background(Color(.purple))
                        .cornerRadius(20)
                    }
                    .sheet(isPresented: $isPickerPresented) {
                        VideoPicker(isPresented: $isPickerPresented, videoURL: $selectedVideoURL)
                    }
                    .onChange(of: selectedVideoURL) { newValue in
                        if let video = newValue {
                            print("Changed Video")
                            processVideo(video)
                            showVideoDisplay = true
                            smoothOutKeyPoints()
                        }
                        else{
                            print("Not changed")
                        }
                    }
                    .background(
                        NavigationLink(destination: VideoPlayerView(videoURL: selectedVideoURL, KeyPointHistory: smoothKeyPoints, selectedSize: selectedSize, videoFrameRate: videoFrameRate), isActive: $showVideoDisplay) {
                            EmptyView()
                        }
                    )
                }
                if isProcessing {
                      ProgressView("Loading Video")
                        .bold()
                  }
            }
            .padding()
            .background(Color.blue)
            .navigationBarHidden(true)
            .sheet(isPresented: $showSettings) {
                SettingsView(useFrontCamera: $useFrontCamera)
            }
            .sheet(isPresented: $showGuidance) {
                GuidanceView()
            }
        }
    }
    
    func processVideo(_ url: URL) {
        isProcessing = true
        processingProgress = 0.0
        KeyPointHistory = [[]]
        smoothKeyPoints = [[]]
        
        let asset = AVAsset(url: url)
        let reader = try! AVAssetReader(asset: asset)
        let videoTrack = asset.tracks(withMediaType: .video).first!
            
        let totalFrames = Int(videoTrack.nominalFrameRate * Float(asset.duration.seconds))
                        
        let readerOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ])
        readerOutput.alwaysCopiesSampleData = false
        reader.add(readerOutput)
        reader.startReading()
        videoFrameRate = Double(videoTrack.nominalFrameRate)
        var processedFrames = 0
        while let sampleBuffer = readerOutput.copyNextSampleBuffer(){
            processedFrames += 1
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                processingProgress = Double(processedFrames) / Double(totalFrames)
            }
            autoreleasepool{
                if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                    // Run your ML model on this pixelBuffer
                    runModel(on: pixelBuffer)
                }
            }
        }
        isProcessing = false
    }
    
    func runModel(on pixelBuffer: CVPixelBuffer) {
        
        guard let uiImage = UIImage(pixelBuffer: pixelBuffer) else { print("Could not capture image"); return }

        detectedBoxes = []
    
        let imageWidth  = uiImage.size.height//Double(CVPixelBufferGetWidth(pixelBuffer))
        let imageHeight = uiImage.size.width//Double(CVPixelBufferGetHeight(pixelBuffer))
        
        selectedSize = CGSize(width: imageWidth, height: imageHeight)
        
        var rotationAngle = vImage.Rotation.clockwise90Degrees
        
        let cgimage = uiImage.cgImage!
        
        let source =  try! vImage.PixelBuffer.makeDynamicPixelBufferAndCGImageFormat(cgImage: cgimage)
        
        // Convert chosen image into 8bit 4 channel RGBA format
        var rgbDestinationImageFormat = vImage_CGImageFormat(
            bitsPerComponent: 8,
            bitsPerPixel: 8 * 4,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)
        )!
        
        let ImgConverter = try! vImageConverter.make(
            sourceFormat: source.cgImageFormat,
            destinationFormat: rgbDestinationImageFormat)
        
        
        let rgbDestinationBuffer = vImage.PixelBuffer<vImage.Interleaved8x4>(
            size: source.pixelBuffer.size)
        
        try! ImgConverter.convert(from: source.pixelBuffer,
                                  to: rgbDestinationBuffer)
        
        
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
        KeyPointHistory.append(detectedKeypoints)
    }
    
    func processImage(_ image: UIImage) {
        
        // Get size of the image
        let imageSize = image.size
        let imageWidth: CGFloat = imageSize.width
        let imageHeight: CGFloat = imageSize.height
        
        selectedSize = imageSize
        
        let img_orientation = image.imageOrientation
        
        var rotationAngle = vImage.Rotation.clockwise90Degrees
        
        switch img_orientation {
        case .up:               rotationAngle = vImage.Rotation.clockwise0Degrees
        case .upMirrored:       rotationAngle = vImage.Rotation.clockwise0Degrees
        case .down:             rotationAngle = vImage.Rotation.clockwise180Degrees
        case .downMirrored:     rotationAngle = vImage.Rotation.clockwise0Degrees
        case .left:             rotationAngle = vImage.Rotation.clockwise270Degrees
        case .leftMirrored:     rotationAngle = vImage.Rotation.clockwise270Degrees
        case .right:            rotationAngle = vImage.Rotation.clockwise90Degrees
        case .rightMirrored:    rotationAngle = vImage.Rotation.clockwise90Degrees
        }
        
        let cgimage = image.cgImage!
        
        // Create dynamic source pixel buffer based on the info of the image chosen
        // Images taken by the camera, sent via Whatsapp all have different bit, alpha value structure.
        let source =  try! vImage.PixelBuffer.makeDynamicPixelBufferAndCGImageFormat(
            cgImage: cgimage)
        //print(source.cgImageFormat.bitmapInfo)
        // Convert chosen image into 8bit 4 channel RGBA format
        let rgbDestinationImageFormat = vImage_CGImageFormat(
            bitsPerComponent: 8,
            bitsPerPixel: 8 * 4,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)
        )!
        
        
        let ImgConverter = try! vImageConverter.make(
            sourceFormat: source.cgImageFormat,
            destinationFormat: rgbDestinationImageFormat)
        
        
        let rgbDestinationBuffer = vImage.PixelBuffer<vImage.Interleaved8x4>(
            size: source.pixelBuffer.size)
        
        try! ImgConverter.convert(from: source.pixelBuffer,
                                  to: rgbDestinationBuffer)
        
        // Now we have converted the buffer into a universal format, we should scale it to the input image size of the ML model
        // Detection model expects image size of (640, 640)
        // Create memory for storing rescaled image
        let resizedBuffer = vImage.PixelBuffer<vImage.Interleaved8x4>(size: vImage.Size(width: 640, height: 640))
        // Scale image
        rgbDestinationBuffer.scale(destination: resizedBuffer)
        
        
        // Rotate RGBDestinationBuffer 90 degrees to the right since for some reason UIImages are stored 90 degrees pre-rotated
        let correctedBuffer = vImage.PixelBuffer<vImage.Interleaved8x4>(
            size: vImage.Size(width: 640, height: 640))
        
        resizedBuffer.rotate(rotationAngle, destination: correctedBuffer)
        
        let modelURL = Bundle.main.url(forResource: "ElifDetModel", withExtension: "mlmodelc")
        
        let visionModel = try! VNCoreMLModel(for: MLModel(contentsOf: modelURL!))
        let recognitions = VNCoreMLRequest(model: visionModel, completionHandler: detectionDidComplete)
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
            
            // Do pose estimation if the bbox is large enough
            let bbox_area = detectedBoxes[i].width * detectedBoxes[i].height
            
            if bbox_area > 0.2 && i == 0{
                
                // Crop the detected person
                // Define region of interest
                let roi_x  = Int(floor(max(detectedBoxes[i].minX * 640 - padding, 0)))
                let roi_y  = Int(floor(640 - max(detectedBoxes[i].minY * 640 - padding, 0)))
                let roi_x2 = Int(floor(min(detectedBoxes[i].maxX * 640 + padding,  640)))
                let roi_y2 = Int(floor(640 - min(detectedBoxes[i].maxY * 640 + padding, 640)))
                
                let roi_w = roi_x2 - roi_x
                let roi_h = roi_y  - roi_y2
                
                let roi = CGRect(x: roi_x, y: roi_y2, width: roi_w, height: roi_h)
                
                croppedBbox = roi
                
                let croppedImageBuffer = correctedBuffer.cropped(to: roi)
                
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
                    if left_leg_keypoints.contains(i) {
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
    func PoseEstimationDidComplete(request: VNRequest, error: Error?) {
        if let pose_results = request.results {
            
            let x_observation = pose_results[0] as! VNCoreMLFeatureValueObservation
            let y_observation = pose_results[1] as! VNCoreMLFeatureValueObservation
            
            let simcc_x = x_observation.featureValue.multiArrayValue!
            let simcc_y = y_observation.featureValue.multiArrayValue!
            
            let left_leg_keypoints = [11, 13, 15, 20, 22, 24]
            let right_leg_keypoints = [12, 14, 16, 21, 23, 25]
            
            let no_keypoints = simcc_x.shape[1].intValue
            
            for i in 0..<no_keypoints {
                
                let x_ptr = UnsafePointer<Float>(OpaquePointer(simcc_x.dataPointer))
                let x_info = argmax(x_ptr.advanced(by: i * 192 * 2), count: 192*2, stride: 1)
                
                let y_ptr = UnsafePointer<Float>(OpaquePointer(simcc_y.dataPointer))
                let y_info = argmax(y_ptr.advanced(by: i * 256 * 2), count: 256*2, stride: 1)
                
                let x_coord = Double(x_info.0) / 2
                let y_coord = Double(y_info.0) / 2
                
                let bbox_x = croppedBbox.minX
                let bbox_y = croppedBbox.minY
                let bbox_w = croppedBbox.width
                let bbox_h = croppedBbox.height
                
                let bbox_xcoord = bbox_x + min(Double(x_coord * bbox_w) / 192.0, bbox_w)
                let bbox_ycoord = bbox_y + min(Double(y_coord * bbox_h) / 256.0, bbox_h)
                
                let resized_xcoord = min(bbox_xcoord / 640 * selectedSize.width, selectedSize.width)
                let resized_ycoord = min(bbox_ycoord / 640 * selectedSize.height, selectedSize.height)
                
                let keypoint = KeyPoint(resized_xcoord, resized_ycoord, Color.red)
                //detectedKeypoints.append(keypoint)
            }
        }
    }
    func convertKeypointsToPoints(_ keypoints: MLMultiArray) -> [KeyPoint] {
        var points: [KeyPoint] = []
        for i in stride(from: 0, to: keypoints.count, by: 2) {
            let x = keypoints[i].doubleValue
            let y = keypoints[i + 1].doubleValue
            points.append(KeyPoint(x, y, Color.red))
        }
        return points
    }
    
    func cropImage(image: UIImage, rect: CGRect, scale: CGFloat) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(CGSize(width: rect.size.width / scale, height: rect.size.height / scale), true, 0.0)
        image.draw(at: CGPoint(x: -rect.origin.x / scale, y: -rect.origin.y / scale))
        let croppedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return croppedImage
    }
    
    func smoothOutKeyPoints() {
        smoothKeyPoints = KeyPointHistory
        
        for i in 1..<KeyPointHistory.count - 1{
            let before = KeyPointHistory[i - 1]
            let after = KeyPointHistory[i + 1]
            let now = KeyPointHistory[i]
            
            if now.count == 6 && before.count == 6 && after.count == 6{
                
                let new0 = KeyPoint(0.8 * now[0].x + 0.1 * after[0].x + 0.1 * before[0].x, 0.8 * now[0].y + 0.1 * after[0].y + 0.1 * before[0].y, now[0].color)

                let new1 = KeyPoint(0.8 * now[1].x + 0.1 * after[1].x + 0.1 * before[1].x, 0.8 * now[1].y + 0.1 * after[1].y + 0.1 * before[1].y, now[1].color)
                
                let new2 = KeyPoint(0.8 * now[2].x + 0.1 * after[2].x + 0.1 * before[2].x, 0.8 * now[2].y + 0.1 * after[2].y + 0.1 * before[2].y, now[2].color)
                
                let new3 = KeyPoint(0.8 * now[3].x + 0.1 * after[3].x + 0.1 * before[3].x, 0.8 * now[3].y + 0.1 * after[3].y + 0.1 * before[3].y, now[3].color)
                
                let new4 = KeyPoint(0.8 * now[4].x + 0.1 * after[4].x + 0.1 * before[4].x, 0.8 * now[4].y + 0.1 * after[4].y + 0.1 * before[4].y, now[4].color)
                
                let new5 = KeyPoint(0.8 * now[5].x + 0.1 * after[5].x + 0.1 * before[5].x, 0.8 * now[5].y + 0.1 * after[5].y + 0.1 * before[5].y, now[5].color)
                
                let news: [KeyPoint] = [new0, new1, new2, new3, new4, new5]
                
                smoothKeyPoints[i] = news
            }
        }
    }
}



struct HomeView_Previews: PreviewProvider {
    static var previews: some View {
        HomeView()
    }
}
