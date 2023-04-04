import cv2
import numpy as np

# Set the block size for motion compensation
block_size = 16

# Open the input video
video = cv2.VideoCapture('xylophone1.avi')

# Get the video properties
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Create a VideoWriter object for the output video
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('video.avi', fourcc, fps, (frame_width, frame_height))

# Loop over the frames
prev_frame = None
while True:
    ret, frame = video.read()

    # Check if the frame is read correctly
    if not ret:
        break

    cv2.imshow('frame1', frame)
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If this is the first frame, write it to the output video and continue
    if prev_frame is None:
        out.write(frame)
        prev_frame = gray_frame
        continue

    # Perform motion-compensated interframe prediction
    prediction = np.zeros_like(gray_frame)
    for j in range(0, gray_frame.shape[0], block_size):
        for i in range(0, gray_frame.shape[1], block_size):
            x = i // block_size
            y = j // block_size

            # Calculate the motion vector for the current block
            min_error = float('inf')
            best_dx = 0
            best_dy = 0
            for dy in range(-8, 9):
                for dx in range(-8, 9):
                    if j+dy < 0 or j+dy+block_size > gray_frame.shape[0] or i+dx < 0 or i+dx+block_size > gray_frame.shape[1]:
                        continue
                    error = np.sum(np.abs(prev_frame[j:j+block_size, i:i+block_size] - gray_frame[j+dy:j+dy+block_size, i+dx:i+dx+block_size]))
                    if error < min_error:
                        min_error = error
                        best_dx = dx
                        best_dy = dy

            # Apply the motion vector to the current block
            prediction[j:j+block_size, i:i+block_size] = gray_frame[j+best_dy:j+best_dy+block_size, i+best_dx:i+best_dx+block_size]

    # Encode the prediction error and write it to the output video
    error = gray_frame - prediction

    cv2.imshow('frame2', error)
    out.write(cv2.cvtColor(error, cv2.COLOR_GRAY2BGR))

    # Set the current frame as the previous frame for the next iteration
    prev_frame = gray_frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video objects and close the output video
video.release()
out.release()
cv2.destroyAllWindows()
