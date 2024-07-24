import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time


def process_video(video_path, output_csv_path, display=False, frame_rate=30, duration=100):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    circle1_positions = []
    circle2_positions = []

    delay = int(1000 / frame_rate)
    start_time = time.time()
    end_time = start_time + duration

    while cap.isOpened() and time.time() < end_time:
        ret, frame = cap.read()

        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detected_positions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    detected_positions.append((cX, cY))

        # Assuming two circles are detected, you need to decide which is which
        if len(detected_positions) >= 2:
            detected_positions = sorted(detected_positions, key=lambda pos: pos[0])
            circle1_positions.append(detected_positions[0])
            circle2_positions.append(detected_positions[1])

            if display:
                for cX, cY in detected_positions:
                    cv2.drawContours(frame, [contour], -1, (0, 0, 0), 2)
                    cv2.circle(frame, (cX, cY), 5, (0, 0, 0), -1)
                    cv2.putText(frame, f"({cX}, {cY})", (cX - 20, cY - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # Optionally, you can add different labels for circle1 and circle2
                cv2.putText(frame, "Circle 1", (circle1_positions[-1][0] - 40, circle1_positions[-1][1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, "Circle 2", (circle2_positions[-1][0] - 40, circle2_positions[-1][1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if display:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    df_circle1 = pd.DataFrame(circle1_positions, columns=['X', 'Y'])
    df_circle1.to_csv(output_csv_path.replace('.csv', '_circle1.csv'), index=False)
    df_circle2 = pd.DataFrame(circle2_positions, columns=['X', 'Y'])
    df_circle2.to_csv(output_csv_path.replace('.csv', '_circle2.csv'), index=False)
    print(f"Circle 1 positions saved to {output_csv_path.replace('.csv', '_circle1.csv')}")
    print(f"Circle 2 positions saved to {output_csv_path.replace('.csv', '_circle2.csv')}")

    for i, (df, label) in enumerate(zip([df_circle1, df_circle2], ["Circle 1", "Circle 2"])):
        df['X_smooth'] = gaussian_filter1d(df['X'], sigma=2)
        df['Y_smooth'] = gaussian_filter1d(df['Y'], sigma=2)

        plt.figure(figsize=(10, 5))
        plt.plot(df['X_smooth'], df['Y_smooth'], color='blue', alpha=0.5, linewidth=1)
        plt.title(f'{label} Positions Over Time (Smoothed)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.hist(df['X'], bins=30, color='blue', alpha=0.7)
        plt.title(f'Histogram of {label} X Positions')
        plt.xlabel('X Position')
        plt.ylabel('Frequency')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.hist(df['Y'], bins=30, color='blue', alpha=0.7)
        plt.title(f'Histogram of {label} Y Positions')
        plt.xlabel('Y Position')
        plt.ylabel('Frequency')
        plt.show()


# Replace the video path with the IP camera URL
video_path = 'http://192.168.1.47:8080/video'
output_csv_path = 'output44.csv'
process_video(video_path, output_csv_path, display=True, frame_rate=30, duration=100)
