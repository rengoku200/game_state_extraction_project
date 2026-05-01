import cv2
import numpy as np
import sys
sys.path.insert(0, "code")
from ui_masks import analyze_kill_feed

test_frames = [
    "data/extracted_frames/kill_feed/t00025.60_f001456.png",  # confirmed TAP13 kill
    "data/extracted_frames/kill_feed/t00026.10_f001484.png",  # confirmed TAP13 + team kill
    "data/extracted_frames/kill_feed/t00026.59_f001512.png",  # confirmed TAP13 + team kill
    "data/extracted_frames/kill_feed/t00088.14_f005012.png",  # false positive — should be False
    "data/extracted_frames/kill_feed/t00082.23_f004676.png",  # empty — should be False
]

for path in test_frames:
    frame = cv2.imread(path)
    if frame is None:
        print(f"Could not load {path}")
        continue
    
    import os
    print(f"\n{os.path.basename(path)}")
    
    # Check actual HSV of this frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    row_white = np.sum(
        cv2.inRange(hsv, 
        np.array([0,0,151]), 
        np.array([179,79,255])), 
        axis=1
    ) / frame.shape[1]
    
    print(f"  Max row white coverage: {row_white.max():.3f}")
    print(f"  Rows above 40% white:  {np.sum(row_white > 0.40)}")
    
    result = analyze_kill_feed(frame)
    print(f"  white_ratio:  {result['white_ratio']}")
    print(f"  self_kill:    {result['self_kill']}")
    print(f"  team_kill:    {result['team_kill']}")
    print(f"  enemy_kill:   {result['enemy_kill']}")
    print(f"  max_consec_white: {result['max_consec_white']}")