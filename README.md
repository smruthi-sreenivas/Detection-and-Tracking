# Object Detection and Tracking

This project demonstrates a method to detect and track a sports ball in a video using YOLOv8 for object detection and KCF (Kernelized Correlation Filters) for object tracking.
![{D5C71D63-0B71-47FC-9B74-A8B7776C3C21}](https://github.com/user-attachments/assets/fd906a5b-370d-4da1-b27d-60e33e3fe297)


## Overview
- **Detection**: Utilizes the YOLOv8 object detection model from the `ultralytics` library to detect objects in video frames. The model specifically filters for "sports ball" objects.
- **Tracking**: Employs the OpenCV legacy `TrackerKCF` to track the detected object across subsequent frames. This ensures computational efficiency by avoiding repeated detection.
- **Fallback Mechanism**: If the tracker fails (e.g., the object moves out of frame or tracking error occurs), the YOLOv8 model re-detects the ball to reinitialize the tracker.

## Key Features
- Real-time object detection and tracking.
- Bounding box visualization for detected and tracked objects.
  - Blue: During detection.
  - Green: During tracking.
  - Red: For text messages (e.g., "Tracking failure detected").
- Displays FPS (Frames Per Second) to monitor performance.
- Saves the processed video with bounding boxes as `output.avi`.

## Requirements
Ensure the following libraries are installed:

- Python 3.x
- OpenCV
- Ultralytics (for YOLOv8)

To install the dependencies, run:
```bash
pip install opencv-python opencv-contrib-python ultralytics
```

## How It Works
1. **Video Input**: The script reads the input video (`soccer-ball.mp4`).
2. **YOLOv8 Detection**: The YOLOv8 model detects objects in the first frame.
3. **Tracker Initialization**: The KCF tracker is initialized using the detected bounding box.
4. **Frame Processing**: For each subsequent frame:
   - The tracker predicts the new location of the object.
   - If tracking fails, YOLOv8 redetects the object, and a new tracker is initialized.
5. **Bounding Box Drawing**: Bounding boxes are drawn based on detection or tracking status.
6. **Output Video**: The processed video is saved as `output.avi`.

## Usage
Run the script using the following command:
```bash
python Detection+Tracking.py
```

### Input Video
Ensure the input video file is named `soccer-ball.mp4` and placed in the same directory as the script. Update the `video_path` variable in the script if the file name or location is different.

### Output
- The processed video with bounding boxes will be saved as `output.avi` in the same directory.
- Real-time display of the video with bounding boxes and FPS.

## Customization
- **Model Configuration**: Replace `yolov8n.pt` with another YOLOv8 model (e.g., `yolov8s.pt`) for higher accuracy or speed.
- **Tracker Type**: Change `tracker_type` to another OpenCV-supported tracker (e.g., `CSRT` or `MIL`) for different tracking performance.

## Limitations
- Tracking may fail under significant occlusion or fast object movements.
- Detection relies on the pretrained YOLOv8 model, which may need fine-tuning for specific datasets or objects.

## References
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Tracking Module](https://docs.opencv.org/master/d9/df8/group__tracking.html)

