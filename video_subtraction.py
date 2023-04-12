import cv2
import numpy as np

cap = cv2.VideoCapture('video2.mp4')

# Read the first frame
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Set a threshold for the contour area
threshold_area = 1000

while True:
    # Read the next frame
    ret, frame2 = cap.read()
    if not ret:
        break # End of video

    # Convert the frame to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Subtract the pixel values of the two frames
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the difference image to obtain a binary mask
    thresh = cv2.threshold(diff, 70, 255, cv2.THRESH_BINARY)[1]

    # Apply any additional image processing operations as needed
    # For example, you could use erosion/dilation to remove noise or fill gaps in the binary mask

    # Find contours in the binary mask
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for contour in contours:
        # Compute the area of the contour
        area = cv2.contourArea(contour)

        # If the area of the contour is greater than the threshold, draw a circle around it
        if area > threshold_area:
            # Compute the center and radius of the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Draw a circle around the contour
            cv2.circle(frame2, center, radius, (0, 0, 255), 2)

            # Add text to the frame
            cv2.putText(frame2, 'Movement Detected', (int(x) - 50, int(y) - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show the resulting frame
    cv2.imshow('Frame', frame2)
    cv2.waitKey(25) # Delay to match video frame rate

    # Set the current frame as the previous frame for the next iteration
    # gray1 = gray2

cap.release()
cv2.destroyAllWindows()
