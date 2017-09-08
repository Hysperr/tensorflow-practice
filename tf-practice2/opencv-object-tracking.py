import cv2
import sys

if __name__ == '__main__':
    # Set up tracker.
    # instead of MIL, you can also use BOOSTING, KCF, TLD, MEDIANFLOW
    tracker = cv2.TrackerKCF_create()
    video = cv2.VideoCapture("C:/Users/garyk/Desktop/walking.mp4")
    # Exit if video not opened
    if not video.isOpened():
        print("Cannot open video")
        sys.exit(1)
    # Read first frame
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        sys.exit(1)
    # Define an initial bounding box x, y, width, height
    bbox = (1530, 450, 350, 850)
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        # Update tracker
        ok, bbox = tracker.update(frame)
        # Draw bounding box
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), thickness=3, lineType=8)  # BGR
        # Display result
        cv2.imshow("Tracking", frame)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
