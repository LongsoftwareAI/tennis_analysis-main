import os
import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    # Check if target file is locked by an external media player
    final_path = output_video_path
    if os.path.exists(output_video_path):
        try:
            with open(output_video_path, 'a') as f:
                pass
        except PermissionError:
            base, ext = os.path.splitext(output_video_path)
            idx = 1
            while True:
                candidate = f"{base}_v{idx}{ext}"
                try:
                    if os.path.exists(candidate):
                        with open(candidate, 'a') as f:
                            pass
                    final_path = candidate
                    break
                except PermissionError:
                    idx += 1
            print(f"[Warning] '{output_video_path}' is currently open in a media player. Saving instead to '{final_path}'")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(final_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    return final_path