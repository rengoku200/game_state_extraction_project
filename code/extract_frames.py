import cv2
import numpy as np
import os



#----
video_pth = "data/raw_vods/tap_13_domination.mov"
output_dir = "data/extracted_frames"
sample_rate = 2
target_width = 1920
target_height = 1080


kill_feed_region = (1460, 40,  420, 220)
ult_charge_region =  (1390, 895, 500, 140)
point_percentage_region = (660,  60,  650, 150)
health_bar_region = (700, 920, 500, 60)


def crop_region(frame, roi):
    """Crop a region of interest from a frame given (x, y, w, h)."""
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]

def extract_frames(video_path, output_dir, sample_rate, target_size):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "kill_feed"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "point_pct"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ult_charge"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "health_bar"), exist_ok=True)


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    
    fps        = cap.get(cv2.CAP_PROP_FPS)
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = int(fps / sample_rate)

    print(f"Video FPS: {fps:.1f} | Total frames: {total} | Saving every {frame_skip} frames")


    frame_idx  = 0
    saved      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # 1. Resize from Retina 2880x1800 → 1920x1080
            frame = cv2.resize(frame, (target_width, target_height))

            # 2. Crop ROIs
            kill_feed = crop_region(frame, kill_feed_region)
            point_pct = crop_region(frame, point_percentage_region)
            ult = crop_region(frame, ult_charge_region)
            health_bar = crop_region(frame, health_bar_region)

            # 3. Save with timestamp-style naming so frames stay ordered
            timestamp = frame_idx / fps
            name      = f"t{timestamp:08.2f}_f{frame_idx:06d}"

            cv2.imwrite(os.path.join(output_dir, "kill_feed", f"{name}.png"), kill_feed)
            cv2.imwrite(os.path.join(output_dir, "point_pct", f"{name}.png"), point_pct)
            cv2.imwrite(os.path.join(output_dir, "ult_charge", f"{name}.png"), ult)
            cv2.imwrite(os.path.join(output_dir, "health_bar", f"{name}.png"), health_bar)

            saved += 1
            if saved % 10 == 0:
                print(f"  Saved {saved} samples (t={timestamp:.1f}s)")

        frame_idx += 1

    cap.release()
    print(f"\nDone. {saved} frame samples saved to {output_dir}/")

if __name__ == "__main__":
    extract_frames(video_pth, output_dir, sample_rate, (target_width, target_height))

