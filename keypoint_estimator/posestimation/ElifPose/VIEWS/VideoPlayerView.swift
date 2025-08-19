//
//  VideoPlayerView.swift
//  ElifPose
//
//  Created by Elif Basokur on 26.08.2024.
//

import SwiftUI
import AVKit

struct VideoPlayerView: View {
    var videoURL: URL?
    var KeyPointHistory: [[KeyPoint]]
    var selectedSize: CGSize
    var videoFrameRate: Double
    @State private var player: AVPlayer?
    @State private var graphData: [CGFloat] = [0] // This will store the graph data
    @State private var currentTime: CGFloat = 0 // This will track the current time of the video
    @State private var smoothKeyPoints: [[KeyPoint]] = []
    
    var body: some View {
        VStack {
            // Video Player View
            VideoPlayer(player: player)
                .onAppear {
                    player = AVPlayer(url: videoURL!)
                    player?.play()
                    setupTimeObserver()
                }
                .scaledToFit()
                .onDisappear {
                    player?.pause()
                }
                .overlay(alignment: .topLeading){
                    GeometryReader{ geometry in
                        ZStack{
                            let path: Path = drawLines(KeyPointHistory[Int(floor(currentTime*videoFrameRate))], to: CGSize(width: geometry.size.width, height: geometry.size.height), originalSize: selectedSize)
                            path
                            .stroke(.gray,  lineWidth: 3)
                            
                            ForEach(scaleKeyPoints(KeyPointHistory[Int(floor(currentTime*videoFrameRate))], to: CGSize(width: geometry.size.width, height: geometry.size.height), originalSize: selectedSize), id: \.self) { keyPoint in
                                Circle()
                                    .fill(keyPoint.color)
                                    .frame(width: 10, height: 10)
                                    .position(keyPoint.cgPoint)
                             }
                            
                        }
                    }
                }
            Spacer()
            // Line Graph View
            LineGraphView(graphData: graphData, currentTime: currentTime, videoFrameRate: videoFrameRate)
                .frame(width: 330, height: 300)
        }
        .onAppear {
            smoothOutKeyPoints()
            generateGraphData()
        }
    }
    
    private func setupTimeObserver() {
        guard let player = player else { return }
        // Observes the player's time every 1/30th of a second (approximate frame rate)
        let interval = CMTime(seconds: 1/videoFrameRate, preferredTimescale: CMTimeScale(NSEC_PER_SEC))
        player.addPeriodicTimeObserver(forInterval: interval, queue: .main) { time in
            let seconds = CMTimeGetSeconds(time)
            currentTime = CGFloat(seconds)
        }
    }
    
    private func smoothOutKeyPoints() {
        smoothKeyPoints = KeyPointHistory
        
        for i in 1..<KeyPointHistory.count - 1{
            let before = KeyPointHistory[i - 1]
            let after = KeyPointHistory[i + 1]
            let now = KeyPointHistory[i]
            
            if now.count == 6 && before.count == 6 && after.count == 6{
                
                let new0 = KeyPoint(0.8 * now[0].x + 0.1 * after[0].x + 0.2 * before[0].x, 0.6 * now[0].y + 0.2 * after[0].y + 0.2 * before[0].y, now[0].color)
                let new1 = KeyPoint(0.6 * now[1].x + 0.2 * after[1].x + 0.2 * before[1].x, 0.6 * now[1].y + 0.2 * after[1].y + 0.2 * before[1].y, now[1].color)
                let new2 = KeyPoint(0.6 * now[2].x + 0.2 * after[2].x + 0.2 * before[2].x, 0.6 * now[2].y + 0.2 * after[2].y + 0.2 * before[2].y, now[2].color)
                let new3 = KeyPoint(0.6 * now[3].x + 0.2 * after[3].x + 0.2 * before[3].x, 0.6 * now[3].y + 0.2 * after[3].y + 0.2 * before[3].y, now[3].color)
                let new4 = KeyPoint(0.6 * now[4].x + 0.2 * after[4].x + 0.2 * before[4].x, 0.6 * now[4].y + 0.2 * after[4].y + 0.2 * before[4].y, now[4].color)
                let new5 = KeyPoint(0.6 * now[5].x + 0.2 * after[5].x + 0.2 * before[5].x, 0.6 * now[5].y + 0.2 * after[5].y + 0.2 * before[5].y, now[5].color)
                let news: [KeyPoint] = [new0, new1, new2, new3, new4, new5]
                smoothKeyPoints[i] = news
            }
        }
    }
    
    
    private func generateGraphData() {
        graphData = []
    
        for i in 0..<KeyPointHistory.count{
            let keypointDetections = KeyPointHistory[i]
            let scaleY = selectedSize.height / 200
            if keypointDetections.count == 6{
                let MTU = findMTU(keypointDetections, to: CGSize(width: 200, height: 200), originalSize: selectedSize)
                graphData.append(MTU / 200)
            }
            else{
                graphData.append(0)
            }
        }
    }
    
    // Scaling key points method
    func scaleKeyPoints(_ keyPoints: [KeyPoint], to viewSize: CGSize, originalSize: CGSize) -> [KeyPoint] {
        return keyPoints.map {
            let scaleY = viewSize.height / originalSize.height
            let offsetX = (viewSize.width - scaleY * originalSize.width) / 2
            return KeyPoint(offsetX + ($0.x) * scaleY,  $0.y * scaleY, $0.color)
        }
    }
    
    func drawLines(_ keyPoints: [KeyPoint], to viewSize: CGSize, originalSize: CGSize) -> Path {
        var path = Path()
        
        if keyPoints.count == 6 {
            
            
            let scaleY = viewSize.height / originalSize.height
            let scaleX = scaleY
            let offsetX = (viewSize.width - scaleY * originalSize.width) / 2
            
            if let firstPoint = keyPoints.first {
                path.move(to: CGPoint(x: (firstPoint.x) * scaleX + offsetX, y: firstPoint.y * scaleY))
            }
            path.addLine(to: CGPoint(x: (keyPoints[0].x) * scaleX + offsetX, y: keyPoints[0].y * scaleY))
            path.addLine(to: CGPoint(x: (keyPoints[1].x) * scaleX + offsetX, y: keyPoints[1].y * scaleY))

            path.addLine(to: CGPoint(x: (keyPoints[2].x) * scaleX + offsetX, y: keyPoints[2].y * scaleY))
            path.addLine(to: CGPoint(x: (keyPoints[5].x) * scaleX + offsetX, y: keyPoints[5].y * scaleY))
            path.addLine(to: CGPoint(x: (keyPoints[3].x) * scaleX + offsetX, y: keyPoints[3].y * scaleY))
            path.move   (to: CGPoint(x: (keyPoints[5].x) * scaleX + offsetX, y: keyPoints[5].y * scaleY))
            path.addLine(to: CGPoint(x: (keyPoints[4].x) * scaleX + offsetX, y: keyPoints[4].y * scaleY))
        }
        
        return path
    }

    func findMTU(_ keyPoints: [KeyPoint], to viewSize: CGSize, originalSize: CGSize) -> CGFloat {
        
        if keyPoints.count == 6 {
            let scaleX = viewSize.width / originalSize.width
            let scaleY = viewSize.height / originalSize.height
            
            // MTU Estimation using Hawkings Hull formula
            
            let xDist: CGFloat    = (keyPoints[1].x * scaleX - keyPoints[2].x * scaleX); //[2]
            let yDist: CGFloat    = (keyPoints[1].y * scaleY - keyPoints[2].y * scaleY); //[3]
            let l_shank: CGFloat = sqrt((xDist * xDist) + (yDist * yDist));
            
            let alpha  = calculateAlpha(keyPoints[0].cgPoint, keyPoints[1].cgPoint, keyPoints[2].cgPoint)
            
            let phi = calculatePhi(keyPoints[1].cgPoint, keyPoints[2].cgPoint, keyPoints[5].cgPoint, keyPoints[3].cgPoint)
            
            let MTU = l_shank * (0.9 + -6.2e-4 * alpha + 2.14e-3 * phi)
            return MTU
        }
        else {
            return 0
        }
        
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

struct LineGraphView: View {
    var graphData: [CGFloat]
    var currentTime: CGFloat
    var videoFrameRate: Double
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Path { path in
                    let stepX = geometry.size.width / CGFloat(graphData.count - 1)
                    let stepY = geometry.size.height - 20
                    
                    path.move(to: CGPoint(x: 12, y: stepY * (1.0 - graphData[0])))
                    
                    for index in 1..<graphData.count {
                        let xPosition = CGFloat(index) * stepX + 12
                        let yPosition = stepY * (1.0 - graphData[index])
                        path.addLine(to: CGPoint(x: xPosition, y: yPosition))
                    }
                }
                .stroke(Color.blue, lineWidth: 2)
                
                let currentX = CGFloat(currentTime*videoFrameRate / CGFloat(graphData.count)) * geometry.size.width
                Path { path in
                    path.move(to: CGPoint(x: currentX + 12, y: geometry.size.height*0.2))
                    path.addLine(to: CGPoint(x: currentX + 12, y: geometry.size.height))
                }
                .stroke(Color.red, lineWidth: 2)
                
                if graphData.count > 1 {
                    let totalTime = Int(Double(graphData.count) / videoFrameRate)
                    let yAxisLabels = stride(from: 0.0, through: 0.8, by: 0.2).map { String(format: "%.1f", $0) }
                    let xAxisLabels = stride(from: 0, through: totalTime, by: totalTime / 4).map { "\($0) s" }
                    ForEach(0..<yAxisLabels.count) { i in
                        let yPosition = geometry.size.height * (1.0 - CGFloat(i) / CGFloat(yAxisLabels.count - 1)) - 20
                        Text(yAxisLabels[i])
                            .font(.caption)
                            .position(x: 10, y: yPosition)
                        Path { path in
                            path.move(to: CGPoint(x: 12, y: yPosition))
                            path.addLine(to: CGPoint(x: geometry.size.width, y: yPosition))
                        }
                        .stroke(Color.gray.opacity(0.5), lineWidth: 1)
                        ForEach(1..<xAxisLabels.count) { i in
                            let xPosition = geometry.size.width * CGFloat(i) / CGFloat(xAxisLabels.count - 1)
                            Path { path in
                                path.move(to: CGPoint(x: xPosition, y: -20))
                                path.addLine(to: CGPoint(x: xPosition, y: geometry.size.height - 20))
                            }
                            .stroke(Color.gray.opacity(0.5), lineWidth: 1)
                            Text(xAxisLabels[i])
                                .font(.caption)
                                .position(x: xPosition, y: geometry.size.height - 20)
                        }
                    }
                }
                
            }
            .padding(10)
        }
    }
}
