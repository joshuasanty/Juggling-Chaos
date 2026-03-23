import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to interpolate points for smoother trails
def interpolate_points(p1, p2, steps=5):
    x_vals = np.linspace(p1[0], p2[0], steps).astype(int)
    y_vals = np.linspace(p1[1], p2[1], steps).astype(int)
    return [(x, y) for x, y in zip(x_vals, y_vals)]

# SCALE_FACTOR = 415.33 #for webcam
SCALE_FACTOR = 2866
video_path = "videos/josh_normal.mp4"  
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video loaded: {video_path}")
print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total Frames: {total_frames}")

whiteLower = (0, 0, 200)
whiteUpper = (180, 50, 255)

trail = []
y_positions = []  # Store y-values of the ball's position
time_steps = []
distance_threshold = 25  # Maximum distance for a point to belong to the same trail

frame_idx = 0
start_time = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break 

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

# Save y-position plot
if len(y_positions) > 0:
    max_y = frame_height
    inverted_y_positions = [(max_y - y) / SCALE_FACTOR for y in y_positions]  # Invert y-data for the plot


    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(time_steps, inverted_y_positions, label="y-position of Ball", color='green')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('y-Position (m)')
    ax1.set_title(f"{video_path.split('/')[-1].split('.')[0].split("_")[-1].capitalize()}" + ' y-Position of Ball Over Time')
    ax1.grid(True)
    ax1.legend()
    fig1.savefig(os.path.join(output_dir, f"{video_path.split('/')[-1].split('.')[0]}_video_y_positions.png"))
    print("y-position plot saved as 'ball_y_positions.png'.")

# Perform Fourier transform
if len(y_positions) > 1:
    y_data = np.array(y_positions)
    fft_result = np.fft.fft(y_data)
    freqs = np.fft.fftfreq(len(y_data), 1 / fps)
    fft_amplitude = np.abs(fft_result)

    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(np.fft.fftshift(freqs), np.fft.fftshift(fft_amplitude), label="Fourier Transform (Amplitude)", color='blue')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f"{video_path.split('/')[-1].split('.')[0].split("_")[-1].capitalize()}" + ' Fourier Transform of y-Position Over Time')
    ax2.grid(True)
    ax2.set_xlim(-5, 5)
    ax2.legend()
    print(f"{video_path.split('\\')[-1].split('.')[0]}_fourier_transform.png")
    fig2.savefig(os.path.join(output_dir, f"{video_path.split('/')[-1].split('.')[0]}_fourier_transform.png"))
    print("Fourier transform plot saved as", f"{video_path.split('/')[-1].split('.')[0]}_fourier_transform.png")

print("Processing complete. Plots saved in 'plots' directory.")
