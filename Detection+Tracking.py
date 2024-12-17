# Import libraries
import cv2

# print(cv2.__version__)
# print("legacy module available:", hasattr(cv2, 'legacy'))

def detection(frame,model):
    global bbox
    # perform object detection on frame
    result = model(frame)
    # check if a ball is present in the result
    # Draw bounding boxes for detected objects
    for result in result[0].boxes.data.tolist():
        #extract bounding box data
        x1, y1, x2, y2, conf, cls = map(int, result[:6])
        label = model.names[cls]
        if label == 'sports ball':  # Filter for sports ball
            color = blue
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} ', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            width = x2 - x1
            height = y2 - y1
            bbox = (x1, y1, width, height)
    return frame,bbox

#load video
video_path = 'soccer-ball.mp4'
video = cv2.VideoCapture(video_path)
# Get video properties
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

#read first frame
ok,frame = video.read()
if not ok:
    print('cannot read video file')

#define colors for bounding boxes
#blue - when performing detection
#green - when tracking
#red - for the text
blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)

#detect ball
#use YOLO from ultralytics
from ultralytics import YOLO
#load the YOLOv8 model
model = YOLO('yolov8n.pt')
frame,bbox = detection(frame,model)

#set up tracker
tracker_type = 'KCF'
tracker = cv2.legacy.TrackerKCF_create()
# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

#go through subsequent frames
while True:
    #read new frame
    ok_read,frame = video.read()
    if not ok_read:
        break
    # Start timer
    timer = cv2.getTickCount()
    # The update method is used to obtain the location of the new tracked object. The method returns
    # false when the track is lost. Tracking can fail because the object went outside the video frame or
    # if the tracker failed to track the object.
    # In both cases, a false value is returned.
    # update tracker
    ok_tracker,bbox = tracker.update(frame)
    # Calculate processing time and display results.
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # Draw bounding box
    if ok_tracker:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2,green , 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, red, 2)
        #detect the ball again
        frame, bbox = detection(frame, model)
        #  print(bbox)
        if bbox != (0.0, 0.0, 0.0, 0.0): #sometimes no bbox is returned
            #initialize a new tracker after tracking fails as old tracker not properly reset with new bounding box
            tracker = cv2.legacy.TrackerKCF_create()
            # Initialize tracker with frame and detected bounding box
            ok_init = tracker.init(frame, bbox)
            if not ok_init:
                print('initialization failure of new tracker')
    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (50, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, blue, 2)
    #display frame
    cv2.imshow('frame',frame)

    # Write the frame to the output video file
    out.write(frame)
    # Exit if the user presses the 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:  # 'Esc' key
        break

video.release()
out.release()
cv2.destroyAllWindows()


#   python Detection+Tracking.py
