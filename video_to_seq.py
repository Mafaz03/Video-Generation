import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from config import IMAGE_SIZE
from tqdm import tqdm

frames_to_capture = 8  # Number of frames to process for each starting point
skip = 5               # Interval to skip frames

import os
videos_1 = os.listdir("video")
for vid_num in tqdm(videos_1):
    videos_2 = "video/" + str(vid_num)
    for vid_num2 in os.listdir(videos_2):
        video_path = videos_2 + "/" + vid_num2

        vidcap = cv2.VideoCapture(video_path)

        if not vidcap.isOpened():
            print("Error: Cannot open video.")
            exit()

        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            print("Error: Video has no frames.")
            exit()

        # Select random start points ensuring enough frames remain
        frame_start = random.choices(range(total_frames - frames_to_capture * skip), k=10)

        for f_idx, f in enumerate(frame_start):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, f)
            canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE * frames_to_capture, 3), dtype=np.uint8)

            captured_frames = 0
            while captured_frames < frames_to_capture:
                ret, frame = vidcap.read()

                # Check if the frame is valid
                if not ret:
                    print(f"Warning: Could not read frame {f}.")
                    break

                # Add the current frame to the canvas
                input_resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
                canvas[:, IMAGE_SIZE * captured_frames:IMAGE_SIZE * (captured_frames + 1), :] = input_resized
                captured_frames += 1

                # Skip frames by setting the video capture position
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, vidcap.get(cv2.CAP_PROP_POS_FRAMES) + skip - 1)

            # Save canvas after processing all frames for this start point
            output_path = f"Dataset/{video_path.split('/')[-1][:-4]}_start_{f}_skip_{skip}.jpg"
            plt.imsave(output_path, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
