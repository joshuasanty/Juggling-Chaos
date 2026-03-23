import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def interpolate_points(p1, p2, steps=5):
    x_vals = np.linspace(p1[0], p2[0], steps).astype(int)
    y_vals = np.linspace(p1[1], p2[1], steps).astype(int)
    return [(x, y) for x, y in zip(x_vals, y_vals)]

def find_velocity_extrema(velocity, time_steps, video_name, output_dir):
    if len(velocity) < 2:
        print(f"Not enough data for peak detection in {video_name}.")
        return

    # Find peaks and valleys
    peaks, _ = find_peaks(velocity, prominence=0.1)
    valleys, _ = find_peaks(-np.array(velocity), prominence=0.5)

    # Extract times and values for peaks and valleys
    peak_times = [time_steps[i] for i in peaks]
    peak_values = [velocity[i] for i in peaks]

    valley_times = [time_steps[i] for i in valleys]
    valley_values = [velocity[i] for i in valleys]

    output_file = os.path.join(output_dir, f"{video_name}_peaks_valleys.txt")
    with open(output_file, "w") as f:
        f.write("Peaks (Throwing Events):\n")
        for idx, (t, v) in enumerate(zip(peak_times, peak_values), start=1):
            f.write(f"{idx}. Time: {t:.2f}s, Velocity: {v:.2f} pixels/s\n")
        f.write("\nValleys (Catching Events):\n")
        for idx, (t, v) in enumerate(zip(valley_times, valley_values), start=1):
            f.write(f"{idx}. Time: {t:.2f}s, Velocity: {v:.2f} pixels/s\n")

    # Save plot of velocity with marked peaks and valleys
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_steps[:len(velocity)], velocity, label="Velocity", color='blue')
    for idx, (t, v) in enumerate(zip(peak_times, peak_values), start=1):
        ax.scatter(t, v, color='red', label="Peaks (Throwing)" if idx == 1 else "", zorder=5)
        ax.text(t, v, f"{idx}", color='red', fontsize=8)
    for idx, (t, v) in enumerate(zip(valley_times, valley_values), start=1):
        ax.scatter(t, v, color='green', label="Valleys (Catching)" if idx == 1 else "", zorder=5)
        ax.text(t, v, f"{idx}", color='green', fontsize=8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(f"{video_name} Velocity Over Time with Peaks and Valleys")
    ax.grid(True)
    ax.legend()
    save_path = os.path.join(output_dir, f"{video_name}_velocity_peaks_valleys.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Velocity peaks and valleys plot saved as '{save_path}'.")

def process_videos(video_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"Processing video: {video_name}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # color range for detecting white balls
        whiteLower = (0, 0, 200)
        whiteUpper = (180, 50, 255)

        SCALE_FACTOR = 2866
        y_positions = []  
        time_steps = []
        velocity = []  

        frame_idx = 0
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
                _, cy = current_position
                y_positions.append(cy)
                time_steps.append(current_time)

                if prev_y is not None:
                    vel = ((cy - prev_y) * fps) / SCALE_FACTOR  # Compute velocity in pixels per second
                    velocity.append(vel)

                prev_y = cy

            frame_idx += 1

        cap.release()
        find_velocity_extrema(velocity, time_steps, video_name, output_dir)

    print("All videos processed. Results saved in output directory.")

video_paths = ["videos/josh_normal.mp4", "videos/josh_fast.mp4", "videos/josh_slow.mp4", "videos/simon_normal.mp4", "videos/simon_fast.mp4", "videos/simon_slow.mp4"]
output_dir = "plots"
process_videos(video_paths, output_dir)
