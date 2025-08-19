# ElifPose iOS App

This folder contains the iOS app for the ElifPose project. The app is designed to perform pose estimation and calculate the Muscle Tendon Length (MTU) of the leg based on user input. Below is a detailed explanation of the app's functionality and the files included in this folder.

## App Functionality

The ElifPose app provides users with three main options for pose estimation and MTU calculation:

### 1. Image Upload
- Users can upload an image, and the app performs pose estimation on the image.
- Keypoints of the leg are displayed on the image, along with the calculated **Muscle Tendon Length (MTU)** of the leg.
- After selecting the first option, users are navigated to `ImagePicker.swift` to choose an image.
- Results are displayed in `ImageDisplayView.swift`, where both keypoints and MTU are shown on the image.

### 2. Video Upload
- Users can upload a video. The app processes the video frame by frame, performing pose estimation and calculating MTU for each frame.
- The results are displayed in `VideoPlayerView.swift`, where users can play the video and view a graph showing MTU at each time point.
- Video selection is handled in `VideoPicker.swift`.

### 3. Real-Time Pose Estimation
- Users can perform real-time pose estimation using the device's camera.
- The app captures video from the camera and displays pose estimation results over the live camera feed.
- This functionality is implemented in `PoseDetectionView.swift`.

## Repo Structure

The ElifPose app source code is organized into logical folders for clarity and maintainability:

```
ElifPose/
├── APP/
│   ├── App.swift
│   ├── Info.plist
│   └── ElifPose.entitlements
├── VIEWS/
│   ├── HomeView.swift
│   ├── GuidanceView.swift
│   ├── ImageDisplayView.swift
│   ├── PoseDetectionView.swift
│   ├── SettingsView.swift
│   └── VideoPlayerView.swift
├── AUX/
│   ├── ImagePicker.swift
│   ├── Keypoint.swift
│   └── VideoPicker.swift
├── UTILS/
│   ├── CGImage+CVPixelBuffer.swift
│   ├── CGImage+RawBytes.swift
│   ├── Math.swift
│   ├── UIImage+CVPixelBuffer.swift
│   └── UIImageExtensions.swift
├── Assets.xcassets
├── Preview Assets.xcassets
├── Preview Content
├── ReadME.md
├──ElifDetModel.mlpackage
├──ElifPoseModel.mlpackage 
```

- **APP/**: Contains the main app entry point and configuration files:
    - `App.swift`: The main entry point for the SwiftUI application.
    - `Info.plist`: App configuration and metadata for iOS.
    - `ElifPose.entitlements`: Entitlements required for app capabilities (e.g., camera, file access).

- **VIEWS/**: All SwiftUI view files that define the user interface and navigation:
    - `HomeView.swift`: The main view users see on app launch, providing navigation to other features.
    - `GuidanceView.swift`: Offers a brief description and guidance on using the app.
    - `ImageDisplayView.swift`: Displays pose estimation results and MTU on uploaded images.
    - `PoseDetectionView.swift`: Handles real-time pose estimation using the device camera.
    - `SettingsView.swift`: Contains settings, including toggling between front and back camera.
    - `VideoPlayerView.swift`: Plays uploaded videos and displays MTU analysis over time.

- **AUX/**: Auxiliary files and helpers:
    - `ImagePicker.swift`: Handles image selection from the device gallery.
    - `Keypoint.swift`: Defines the keypoint class used for pose estimation.
    - `VideoPicker.swift`: Handles video selection from the device gallery.

- **UTILS/**: Utility files and extensions for image processing and math:
    - `CGImage+CVPixelBuffer.swift`: Converts between CGImage and CVPixelBuffer formats.
    - `CGImage+RawBytes.swift`: Provides raw image byte access utilities.
    - `Math.swift`: Implements mathematical operations used throughout the app.
    - `UIImage+CVPixelBuffer.swift`: Converts between UIImage and CVPixelBuffer.
    - `UIImageExtensions.swift`: Additional UIImage helper functions.

- **Assets.xcassets, Preview Assets.xcassets, Preview Content**: Asset and preview resources for the app, including images and UI previews.
- **ElifDetModel.mlpackage, ElifPoseModel.mlpackage**: CoreML models used for detection and pose estimation.

## How to Use the App

1. Open the app, and choose from the three available options:
   - **Image Upload**
   - **Video Upload**
   - **Real-Time Pose Estimation**
2. For image or video upload, follow the prompts to select a file.
3. View the results, which include keypoints and Muscle Tendon Length (MTU) for the leg.
4. For real-time estimation, ensure the camera is enabled, and the results will be displayed live.