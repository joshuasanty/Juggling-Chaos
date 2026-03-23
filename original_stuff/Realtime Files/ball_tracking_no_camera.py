import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

whiteLower = (0, 0, 200)
whiteUpper = (180, 50, 255)

# Initialize trails for each ball
trails = [[], [], []]  # Separate trail lists for up to 3 balls
max_trail_length = 15  # 0.5 seconds at 30 fps
distance_threshold = 25  # Maximum distance for a point to belong to the same trail

canvas_height, canvas_width = 480, 640
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
framerate = 30
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

prev_y = 0
prev_time = time.time()
plt.ion()  # interactive mode
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('y displacement')
ax1.set_ylabel('y-velocity (pixels/s)')
ax1.set_title('Phase-Space Plot: y-Displacement vs. y-Velocity')
ax1.grid(True)

phase_space_points = []
max_points = 50  # Keep the last N points

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    time_diff = current_time - prev_time
    prev_time = current_time

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, whiteLower, whiteUpper)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    current_positions = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            current_positions.append((cx, cy))

    canvas.fill(0)

    for i in range(len(trails)):
        if i < len(current_positions):
            cx, cy = current_positions[i]
            current_y = canvas_height - cy
            v_y = (current_y - prev_y) / time_diff
            prev_y = current_y

            # Update phase space points
            phase_space_points.append((current_y, v_y))
            if len(phase_space_points) > max_points:
                phase_space_points = phase_space_points[-max_points:]

            # Extract x and y coordinates
            x_vals, y_vals = zip(*phase_space_points)

            # Update the plot
            ax1.clear()
            ax1.plot(x_vals, y_vals, '-o', c='b', markersize=5, label='Phase Trajectory')
            ax1.set_xlabel('y displacement')
            ax1.set_ylabel('y-velocity (pixels/s)')
            ax1.set_title('Phase-Space Plot: y-Displacement vs. y-Velocity')
            ax1.grid(True)
            ax1.legend()

            # Update the figure
            fig.canvas.draw()
            fig.canvas.flush_events()

            cv2.circle(canvas, (cx, cy), 10, (0, 255, 0), -1)

    cv2.imshow('Tracking Balls with Smooth Trajectories', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
