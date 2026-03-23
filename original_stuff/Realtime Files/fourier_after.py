import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Start timing
start_time = time.time()

print("Initializing webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"Webcam initialized in {time.time() - start_time:.2f} seconds.")

# Set the camera to use MJPEG format
print("Configuring webcam settings...")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Verify resolution settings
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution set to {int(width)}x{int(height)}")

whiteLower = (0, 0, 200)
whiteUpper = (180, 50, 255)

# Initialize trail for the single ball
trail = []
y_positions = []  # Store y-values of the ball's position
max_trail_length = 15  # 0.5 seconds at 30 fps
distance_threshold = 25  # Maximum distance for a point to belong to the same trail
time_step = 0.033  # Approx. 30 fps, constant time step in seconds

canvas_height, canvas_width = 480, 640
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

print("Starting main loop...")

def interpolate_points(p1, p2, steps=5):
    """
    Interpolate between two points with a specified number of steps.
    Args:
        p1 (tuple): Starting point (x1, y1).
        p2 (tuple): Ending point (x2, y2).
        steps (int): Number of interpolated points.
    Returns:
        list: List of interpolated points.
    """
    x_vals = np.linspace(p1[0], p2[0], steps).astype(int)
    y_vals = np.linspace(p1[1], p2[1], steps).astype(int)
    return [(x, y) for x, y in zip(x_vals, y_vals)]

# Set up real-time plotting
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# First plot: y-position of the ball
line1, = ax1.plot([], [], label="y-position of Ball", color='green')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('y-Position (pixels)')
ax1.set_title('y-Position of Ball Over Time')
ax1.grid(True)
ax1.legend()

# Second plot: Fourier transform of the y-position (won't update in real-time)
line2, = ax2.plot([], [], label="Fourier Transform (Amplitude)", color='blue')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Amplitude')
ax2.set_title('Fourier Transform of y-Position Over Time')
ax2.grid(True)
ax2.legend()

# Create initial empty arrays for plotting
time_steps = []
frame_count = 0
loop_start_time = time.time()

try:
    while True:
        # Measure frame capture time
        frame_start_time = time.time()

        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_count += 1
        if frame_count == 1:
            print(f"First frame captured after {time.time() - loop_start_time:.2f} seconds.")

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for white objects
        mask = cv2.inRange(hsv, whiteLower, whiteUpper)

        # Apply a Gaussian blur to the mask to smooth it out
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # Find contours (balls) in the masked image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours based on area and keep only the largest one
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        current_position = None
        for contour in contours:
            # Get the center of the contour (ball position)
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                current_position = (cx, cy)

        if current_position:
            cx, cy = current_position

            # Record y-position
            y_positions.append(cy)
            time_steps.append(time_step * frame_count)

            # Update trail
            if len(trail) > 0:
                last_point = trail[-1]
                distance = np.sqrt((cx - last_point[0]) ** 2 + (cy - last_point[1]) ** 2)
                if distance < distance_threshold:
                    interpolated_points = interpolate_points(last_point, (cx, cy))
                    trail.extend(interpolated_points)
                else:
                    trail = [(cx, cy)]  # Start new trail if distance is too far
            else:
                trail.append((cx, cy))

            # Limit the trail length
            if len(trail) > max_trail_length:
                trail = trail[-max_trail_length:]

            # Draw the trail on the canvas
            canvas.fill(0)  # Clear the canvas
            for j in range(1, len(trail)):
                cv2.line(canvas, trail[j - 1], trail[j], (0, 255, 0), 2)
            cv2.circle(canvas, trail[-1], 10, (0, 255, 0), -1)

        # Display the canvas
        cv2.imshow('Tracking Ball with Smooth Trajectories', canvas)
        # Get the maximum y-coordinate for inversion
        max_y = canvas_height

        # Update the y-position plot
        if len(y_positions) > 0:
            inverted_y_positions = [max_y - y for y in y_positions]  # Invert y-data for the plot

            line1.set_xdata(time_steps)
            line1.set_ydata(inverted_y_positions)  # Use the inverted y-data
            ax1.relim()  # Recalculate limits
            ax1.autoscale_view()  # Rescale the view
            plt.draw()
            plt.pause(0.01)  # Pause briefly to update the plot

except KeyboardInterrupt:
    print("Program interrupted.")

finally:
    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

    # Perform Fourier Transform after loop ends
    if len(y_positions) > 1:
        y_data = np.array(y_positions)
        fft_result = np.fft.fft(y_data)
        freqs = np.fft.fftfreq(len(y_data), time_step)
        fft_amplitude = np.abs(fft_result)

        # Save the plots as images
        print("Saving the plots...")

        # Save the y-position plot
        fig1 = plt.figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        ax1.plot(time_steps, y_positions, label="y-position of Ball", color='green')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('y-Position (pixels)')
        ax1.set_title('y-Position of Ball Over Time')
        ax1.grid(True)
        ax1.legend()
        fig1.savefig('ball_y_positions.png')  # Save y-position plot
        print("y-position plot saved as 'ball_y_positions.png'.")

        # Save the Fourier Transform plot
        fig2 = plt.figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        ax2.plot(np.fft.fftshift(freqs), np.fft.fftshift(fft_amplitude), label="Fourier Transform (Amplitude)", color='blue')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Fourier Transform of y-Position Over Time')
        ax2.grid(True)
        ax2.legend()
        fig2.savefig('fourier_transform.png')  # Save Fourier Transform plot
        print("Fourier transform plot saved as 'fourier_transform.png'.")

        # Show completion message
        print("All plots saved successfully.")
