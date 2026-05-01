import cv2
import sys
import os
sys.path.insert(0, "code")
from ui_masks import analyze_kill_feed

# Test on several frames including ones where you know kills happened
test_frames = [
    "data/extracted_frames/kill_feed/t00016.25_f000924.png",  # ~11:11
    "data/extracted_frames/kill_feed/t00020.19_f001148.png",  # ~11:15
    "data/extracted_frames/kill_feed/t00034.47_f001960.png",  # ~11:29
    "data/extracted_frames/kill_feed/t00096.51_f005488.png",  # known kill frame
]

for path in test_frames:
    frame = cv2.imread(path)
    if frame is None:
        print(f"Could not load {path}")
        continue
    result = analyze_kill_feed(frame)
    print(f"\n{os.path.basename(path)}")
    print(f"  white_ratio:  {result['white_ratio']}")
    print(f"  blue_ratio:   {result['blue_ratio']}")
    print(f"  yellow_ratio: {result['yellow_ratio']}")
    print(f"  self_kill:    {result['self_kill']}")
    print(f"  team_kill:    {result['team_kill']}")
    print(f"  enemy_kill:   {result['enemy_kill']}")