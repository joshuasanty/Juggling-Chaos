import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# Function to interpolate points for smoother trails
def interpolate_points(p1, p2, steps=5):
    x_vals = np.linspace(p1[0], p2[0], steps).astype(int)
    y_vals = np.linspace(p1[1], p2[1], steps).astype(int)
    return [(x, y) for x, y in zip(x_vals, y_vals)]

video_path = "videos/josh_normal.mp4" 
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video loaded: {video_path}")
print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total Frames: {total_frames}")

# Define color range for detecting white balls
whiteLower = (0, 0, 200)
whiteUpper = (180, 50, 255)

SCALE_FACTOR = 2866
trail = []
y_positions = [] 
time_steps = []
velocity = []  
distance_threshold = 25  # Maximum distance for a point to belong to the same trail

frame_idx = 0
start_time = 0.0
prev_y = None

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    current_time = frame_idx / fps

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, whiteLower, whiteUpper)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    # Find contours (balls)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    current_position = None
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            current_position = (cx, cy)

    if current_position:
        cx, cy = current_position
        y_positions.append(cy)
        time_steps.append(current_time)

        if prev_y is not None:
            vel = ((cy - prev_y) * fps)/SCALE_FACTOR  # Compute velocity in pixels per second
            velocity.append(vel)

        prev_y = cy

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

    frame_idx += 1

cap.release()
# print(velocity)
# Save phase-space plot

# Peak finding for velocity extrema
def find_velocity_extrema(velocity, time_steps):
    if len(velocity) < 2:
        print("Not enough data for peak detection.")
        return

    # Find peaks and valleys
    peaks, _ = find_peaks(velocity, prominence=0.1)
    valleys, _ = find_peaks(-np.array(velocity), prominence=0.5)

    # Extract times and values for peaks and valleys
    peak_times = [time_steps[i] for i in peaks]
    peak_values = [velocity[i] for i in peaks]

    valley_times = [time_steps[i] for i in valleys]
    valley_values = [velocity[i] for i in valleys]

    print("Peaks (Throwing Events):")
    for t, v in zip(peak_times, peak_values):
        print(f"Time: {t:.2f}s, Velocity: {v:.2f} pixels/s")

    print("Valleys (Catching Events):")
    for t, v in zip(valley_times, valley_values):
        print(f"Time: {t:.2f}s, Velocity: {v:.2f} pixels/s")

    # Save plot of velocity with marked peaks and valleys
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_steps[:len(velocity)], velocity, label="Velocity", color='blue')
    ax.scatter(peak_times, peak_values, color='red', label="Peaks (Throwing)", zorder=5)
    ax.scatter(valley_times, valley_values, color='green', label="Valleys (Catching)", zorder=5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(f"{video_path.split('/')[-1].split('.')[0].split("_")[-1].capitalize()}" + ' Velocity Over Time with Peaks and Valleys')
    ax.grid(True)
    ax.legend()
    save_path = os.path.join(output_dir, f"{video_path.split('/')[-1].split('.')[0]}_velocity_peaks_valleys.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Velocity peaks and valleys plot saved as '{save_path}'.")

find_velocity_extrema(velocity, time_steps)

print("Processing complete. Plots saved in 'plots' directory.")
