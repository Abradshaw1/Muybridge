import SwiftUI

struct GuidanceView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Welcome to ElifPose!")
                    .font(.title)
                    .bold()
                    .padding(.bottom, 10)

                Text("Welcome to ElifPose, your companion for real-time pose detection and analysis. This app uses advanced machine learning models to detect and analyze keypoints on your body, helping you monitor your movements and improve your posture.")
                    .font(.body)
                    .fixedSize(horizontal: false, vertical: true)

                sectionHeader("How to Use ElifPose")

                instructionStep(number: 1, title: "Detect Pose in Real-Time:", description: "Navigate to the HomeView and tap on the \"Detect Pose\" button. Allow the app to access your camera if prompted. Ensure you are in a well-lit area for the best results. The app will display keypoints detected on your body in real-time. Move around to see how the keypoints track your movements.")

                instructionStep(number: 2, title: "Upload and Analyze an Image:", description: "Tap the \"Upload\" button on the HomeView. Select a photo from your gallery that you want to analyze. Ensure the photo clearly shows your full body for accurate keypoint detection. The app will process the image and display the detected keypoints.")

                instructionStep(number: 3, title: "Settings:", description: "Access the settings by tapping the gear icon on the HomeView. Toggle between the front and back camera depending on your preference. Adjust other settings to customize your experience.")

                instructionStep(number: 4, title: "Guidance:", description: "Tap the book icon on the HomeView for more tips and information about using the app.")

                sectionHeader("Understanding Pose Detection")

                Text("ElifPose uses two advanced machine learning models to detect and analyze body keypoints:")
                    .font(.body)
                    .fixedSize(horizontal: false, vertical: true)

                bulletPoint(title: "RTMDet Model:", description: "This model is responsible for detecting the presence and bounding boxes of objects (in this case, the human body) in the camera feed.")

                bulletPoint(title: "RTMPose Model:", description: "Once the bounding box is identified, this model analyzes the specific keypoints on your body, such as your joints and limbs, to provide detailed pose information.")

                Text("These models were originally trained using PyTorch and then converted to CoreML for efficient performance on iOS devices. CoreML enables the models to run directly on your device, ensuring real-time performance without needing an internet connection.")
                    .font(.body)
                    .fixedSize(horizontal: false, vertical: true)

                sectionHeader("Tips for Best Results")

                bulletPoint(title: "Lighting:", description: "Ensure you are in a well-lit environment. Poor lighting can affect the accuracy of keypoint detection.")

                bulletPoint(title: "Background:", description: "A plain background helps the models focus on your body without distractions.")

                bulletPoint(title: "Distance:", description: "Position yourself at an optimal distance from the camera to ensure your entire body fits within the frame.")

                bulletPoint(title: "Clothing:", description: "Wear clothing that contrasts with the background to help the models detect your keypoints more accurately.")

                sectionHeader("About the Technology")

                Text("ElifPose leverages state-of-the-art computer vision and deep learning technologies to deliver real-time pose detection. The app's core is built on SwiftUI, providing a responsive and user-friendly interface. The machine learning models are integrated using CoreML, Apple's framework for running machine learning models on iOS devices.")
                    .font(.body)
                    .fixedSize(horizontal: false, vertical: true)

                sectionHeader("Future Improvements")

                Text("We are continually working to improve ElifPose. Here are some features we are planning to add in the future:")
                    .font(.body)
                    .fixedSize(horizontal: false, vertical: true)

                bulletPoint(title: "Pose Correction Suggestions:", description: "Real-time feedback on how to adjust your posture for better alignment.")

                bulletPoint(title: "Activity Tracking:", description: "Detailed logs of your movements and poses over time.")

                bulletPoint(title: "Enhanced Accuracy:", description: "Ongoing improvements to the machine learning models for more precise keypoint detection.")

                sectionHeader("Feedback and Support")

                Text("Your feedback is invaluable to us. If you have any suggestions, encounter issues, or need support, please contact us at [support@elifpose.com](mailto:support@elifpose.com).")
                    .font(.body)
                    .fixedSize(horizontal: false, vertical: true)

                Text("Thank you for using ElifPose! We hope it helps you achieve your fitness and posture goals.")
                    .font(.body)
                    .padding(.bottom, 20)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding()
        }
        .navigationTitle("User Guidance")
    }

    func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.headline)
            .bold()
            .padding(.vertical, 10)
            .fixedSize(horizontal: false, vertical: true)
    }

    func instructionStep(number: Int, title: String, description: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("\(number). \(title)")
                .font(.subheadline)
                .bold()
                .fixedSize(horizontal: false, vertical: true)
            Text(description)
                .font(.body)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.bottom, 10)
    }

    func bulletPoint(title: String, description: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("• \(title)")
                .font(.subheadline)
                .bold()
                .fixedSize(horizontal: false, vertical: true)
            Text(description)
                .font(.body)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.bottom, 10)
    }
}

struct UserGuidanceView_Previews: PreviewProvider {
    static var previews: some View {
        GuidanceView()
    }
}
