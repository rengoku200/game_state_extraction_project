import cv2
import numpy as np
import os
import csv
import sys
import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



from ui_masks import (
    analyze_kill_feed,
    analyze_domination,
    analyze_health,
    analyze_ult,
    apply_hsv_mask,
)
from template_match import (
    load_templates,
    detect_heroes_in_killfeed,
    classify_kill_side,
)

#--------------------------------------


kill_feed_directory = "data/extracted_frames/kill_feed"
point_pct_directory = "data/extracted_frames/point_pct"
health_directory    = "data/extracted_frames/health_bar"
ult_directory       = "data/extracted_frames/ult_charge"
templates_directory = "data/templates"
results_directory   = "results"

# 1 second gap between kill events to capture multiple instances within the feed
kill_cooldown_frames = 2

# Addressing the pov switch when the main player dies and previews a kill cam 
kill_cam_duration_frames = 8




def parse_timestamp(filename):
    """
    Extract timestamp in seconds from frame filename.
    e.g. t00096.51_f005488.png → 96.51
    """
    basename = os.path.basename(filename)
    t_part   = basename.split("_")[0]
    seconds  = float(t_part.replace("t", ""))
    return seconds



def get_sorted_frames(directory):
    """
    Return sorted list of full paths to all PNG frames in a directory.
    Sorted by filename ensures chronological order.
    """
    files = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(".png")
    ]
    return files


def compute_event_weight(event):
    """
    Assign a turning point weight to a kill event based on game context.
    Higher weight = more impactful moment in the game.

    Base weights:
    - TAP13 self kill       → +3  (streamer got the kill)
    - teammate kill         → +1  (ally got the kill)
    - enemy kill            → -1  (bad for our team)

    Bonus weights:
    - enemy dom > 80%       → +2  (kill while enemy close to winning)
    - health < 30%          → +2  (low health clutch kill)
    - ult ready at kill     → +1  (ult available context)
    - team dom > 80%        → +1  (kill while team is close to winning)
    """
    weight = 0

    kill_type  = event.get("kill_type")
    enemy_fill = event.get("enemy_fill", 0)
    team_fill  = event.get("team_fill",  0)
    health_pct = event.get("health_pct", 1.0)
    ult_ready  = event.get("ult_ready",  False)




    #Base weight from kill type
    if kill_type == "self_kill":
        weight += 3
    elif kill_type == "team_kill":
        weight += 1
    elif kill_type == "enemy_kill":
        weight -= 1

    # Bonus weight for clutch moments based of percentage 
    if enemy_fill > 0.8:
        weight += 2
    if team_fill > 0.8:
        weight += 1
    if health_pct < 0.3:
        weight += 2
    if ult_ready:
        weight += 1

    return weight




def detect_pov_switch(kill_data, idx, last_tap13_death_frame):
    """
    Detect if we are currently in a kill cam (POV switched to enemy).

    Logic:
    - When TAP13 dies (enemy_kill with high white ratio) flag the death frame
    - Mark the next KILLCAM_DURATION_FRAMES frames as kill cam
    - Once kill cam window passes, restore normal POV

    Returns (is_killcam, updated_last_death_frame)
    """
    # Detect TAP13 death — enemy kill with significant white flash
    # The white flash from TAP13 dying is distinct
    tap13_died = (
        kill_data["enemy_kill"] and
        kill_data["white_ratio"] > 0.05
    )

    if tap13_died:
        last_tap13_death_frame = idx


    is_killcam = (
        last_tap13_death_frame >= 0 and
        (idx - last_tap13_death_frame) <= kill_cam_duration_frames and
        not tap13_died  # don't flag the death frame itself as killcam
    )

    return is_killcam, last_tap13_death_frame

#Deep learning detection
def load_cnn_classifier(weights_path="results/icon_classifier/best.pt"):
    """Loads the trained ResNet18 model into memory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    
    # Rebuild the ResNet18 architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model"])
    model.eval() # Set to evaluation mode (turns off dropout/batchnorm)
    model.to(device)
    
    # The exact same transforms from your training script
    val_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, classes, device, val_transforms

def classify_crop_cnn(crop_bgr, model, classes, device, transform, threshold=0.60):
    """Feeds a tiny 50x24 image crop into the CNN and returns the hero name."""
    # Convert OpenCV (BGR) to PyTorch Image (RGB)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    
    # Apply transforms and add batch dimension [1, C, H, W]
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        # Convert raw logits to percentages (0.0 to 1.0)
        probs = F.softmax(logits, dim=1)
        max_prob, pred_idx = torch.max(probs, 1)
        
    # THE SHIELD RESTORER: If the CNN is confused, force it to output "unknown"
    if max_prob.item() < threshold:
        return "unknown"
        
    return classes[pred_idx.item()]

def detect_heroes_cnn(kill_crop, model, classes, device, transform):
    """Mimics your old template matcher by slicing the lanes and calling the CNN."""
    h, w, _ = kill_crop.shape
    
    # Lane Locking: Left 35% is Killer, Right 35% is Victim
    killer_crop = kill_crop[:, :int(w*0.35)]
    victim_crop = kill_crop[:, int(w*0.65):]
    
    killer_name = classify_crop_cnn(killer_crop, model, classes, device, transform)
    victim_name = classify_crop_cnn(victim_crop, model, classes, device, transform)
    
    # Return in the exact dictionary format your pipeline expects
    return [
        {"hero": killer_name, "side": "killer"},
        {"hero": victim_name, "side": "victim"}
    ]


def run_pipeline(mode="heuristic"):
    os.makedirs(results_directory, exist_ok=True)

    # Load hero templates
    print("Loading templates...")
    templates = load_templates(templates_directory)
    print(f"Loaded {len(templates)} templates\n")

    if mode == "cnn":
        print("Loading Deep Learning Classifier...")
        cnn_model, cnn_classes, cnn_device, cnn_transform = load_cnn_classifier()
        print(f"Loaded ResNet18 capable of identifying {len(cnn_classes)} heroes\n")

    # Get sorted frame lists for all 4 ROIs
    kill_frames   = get_sorted_frames(kill_feed_directory)
    point_frames  = get_sorted_frames(point_pct_directory)
    health_frames = get_sorted_frames(health_directory)
    ult_frames    = get_sorted_frames(ult_directory)

    total_frames = len(kill_frames)
    print(f"Processing {total_frames} frames...\n")


    #CSV writers 
    continuous_path = os.path.join(results_directory, "continuous.csv")
    continuous_fields = [
        "timestamp", "frame_idx",
        "health_pct", "overshield",
        "enemy_fill", "team_fill",
        "enemy_won_point", "team_won_point",
        "ult_ready", "ult_active",
        "ability_on_cooldown",
        "is_killcam",           # flag for kill cam frames
        "pov",                 
    ]


    events_path = os.path.join(results_directory, "events.csv")
    events_fields = [
        "timestamp", "frame_idx",
        "kill_type",
        "killer_hero", "victim_hero",
        "health_pct_at_kill",
        "enemy_fill_at_kill",
        "team_fill_at_kill",
        "ult_ready_at_kill",
        "event_weight",
        "is_killcam",
    ]


   # State tracking variables
    last_kill_frame        = -kill_cooldown_frames
    last_tap13_death_frame = -kill_cam_duration_frames - 1
    events_buffer          = []
    killcam_count          = 0
    kill_count             = 0
    last_kill_signature    = ("none", "none") # NEW: Tracks who just died

    with open(continuous_path, "w", newline="") as cont_f, \
         open(events_path,     "w", newline="") as evt_f:

        cont_writer = csv.DictWriter(cont_f, fieldnames=continuous_fields)
        evt_writer  = csv.DictWriter(evt_f,  fieldnames=events_fields)

        cont_writer.writeheader()
        evt_writer.writeheader()

        for idx in range(total_frames):

            # Load all 4 ROI crops for this frame
            kill_crop   = cv2.imread(kill_frames[idx])
            point_crop  = cv2.imread(point_frames[idx])
            health_crop = cv2.imread(health_frames[idx])
            ult_crop    = cv2.imread(ult_frames[idx])

            if any(c is None for c in [kill_crop, point_crop, health_crop, ult_crop]):
                print(f"  Skipping frame {idx} — could not load all crops")
                continue

            timestamp = parse_timestamp(kill_frames[idx])

            #Run analyzers
            kill_data   = analyze_kill_feed(kill_crop)
            point_data  = analyze_domination(point_crop)
            health_data = analyze_health(health_crop)
            ult_data    = analyze_ult(ult_crop)
            

            #POV / kill cam detection
            is_killcam, last_tap13_death_frame = detect_pov_switch(
                kill_data, idx, last_tap13_death_frame
            )
            pov = "killcam" if is_killcam else "tap13"
            if is_killcam:
                killcam_count += 1

            

            cont_writer.writerow({
                "timestamp":           round(timestamp, 2),
                "frame_idx":           idx,
                "health_pct":          health_data["health_pct"],
                "overshield":          health_data["overshield"],
                "enemy_fill":          point_data["enemy_fill"],
                "team_fill":           point_data["team_fill"],
                "enemy_won_point":     point_data["enemy_won_point"],
                "team_won_point":      point_data["team_won_point"],
                "ult_ready":           ult_data["ult_ready"],
                "ult_active":          ult_data["ult_active"],
                "ability_on_cooldown": ult_data["ability_on_cooldown"],
                "is_killcam":          is_killcam,
                "pov":                 pov,
            })


            #Kill event detection while also skipping the pov switch to enemy 
           # Kill event detection while also skipping the pov switch to enemy 
            kill_detected = (
                kill_data["self_kill"] or
                kill_data["team_kill"] or
                kill_data["enemy_kill"]
            )

            if kill_detected and not is_killcam:
                # THE COMMAND SWITCH
                if mode == "cnn":
                    detections = detect_heroes_cnn(kill_crop, cnn_model, cnn_classes, cnn_device, cnn_transform)
                else:
                    # Fallback to your 17/18 Template Matcher
                    detections = detect_heroes_in_killfeed(kill_crop, templates)
                
                killer_hero = "unknown"
                victim_hero = "unknown"

                # 1. Assign names (with Psylocke override)
                if kill_data["self_kill"]:
                    killer_hero = "psylocke"
                
                for det in detections:
                    name = det["hero"].replace("_normal", "").replace("_mirrored", "")
                    # TIE-BREAKER: Favor Invisible Woman over White Fox if it's early game
                    if name == "whitefox_lord" and timestamp < 100: # Adjust time as needed
                        name = "inviswomen_lord"

                    if det["side"] == "killer" and killer_hero == "unknown":
                        killer_hero = name
                    elif det["side"] == "victim" and victim_hero == "unknown":
                        victim_hero = name

                # 2. THE HEAVY-DUTY SHIELD:
                # If we logged a kill in the last 4 seconds, and EITHER name matches
                # (or is unknown), it is the SAME kill. Block it.
                current_signature = (killer_hero, victim_hero)
                is_duplicate = False
                
                if last_kill_signature != ("none", "none"):
                    last_k, last_v = last_kill_signature
                    
                    # 1. Exact Duplicate: Same killer and same victim within 5 seconds
                    if killer_hero == last_k and victim_hero == last_v:
                        if (idx - last_kill_frame) < 10: 
                            is_duplicate = True
                    
                    # 2. Flicker Duplicate: Both unknown within 3 seconds
                    elif killer_hero == "unknown" and victim_hero == "unknown":
                        if (idx - last_kill_frame) < 6: 
                            is_duplicate = True

                # 3. THE UPGRADE OVERRIDE: 
                # If we were going to skip this as a duplicate, but this frame actually 
                # found a hero name where the last one was 'unknown', LOG IT ANYWAY!
                if is_duplicate:
                    found_new_victim = (victim_hero != "unknown" and last_v == "unknown")
                    found_new_killer = (killer_hero != "unknown" and last_k == "unknown")
                    
                    if found_new_victim or found_new_killer:
                        is_duplicate = False # Force a new log with the better data

                # --- 3. LOGGING ---
                if not is_duplicate:
                    last_kill_frame = idx
                    last_kill_signature = current_signature # Update memory for next time
                    kill_count += 1

                    # Determine kill type
                    if kill_data["self_kill"]:
                        kill_type = "self_kill"
                    elif kill_data["team_kill"]:
                        kill_type = "team_kill"
                    else:
                        kill_type = "enemy_kill"

                    # Event dict with every variable present 
                    event = {
                        "timestamp":          round(timestamp, 2),
                        "frame_idx":          idx,
                        "kill_type":          kill_type,
                        "killer_hero":        killer_hero,
                        "victim_hero":        victim_hero,
                        "health_pct_at_kill": health_data["health_pct"],
                        "enemy_fill_at_kill": point_data["enemy_fill"],
                        "team_fill_at_kill":  point_data["team_fill"],
                        "ult_ready_at_kill":  ult_data["ult_ready"],
                        "event_weight":       compute_event_weight({
                            "kill_type":  kill_type,
                            "enemy_fill": point_data["enemy_fill"],
                            "team_fill":  point_data["team_fill"],
                            "health_pct": health_data["health_pct"],
                            "ult_ready":  ult_data["ult_ready"],
                        }),
                        "is_killcam": is_killcam,
                    }

                    evt_writer.writerow(event)
                    events_buffer.append(event)

                    print(f"  [{timestamp:.1f}s] {kill_type:12s} | "
                          f"{killer_hero:20s} → {victim_hero:20s} | "
                          f"weight={event['event_weight']:+d} | "
                          f"health={health_data['health_pct']:.2f} | "
                          f"enemy%={point_data['enemy_fill']:.2f} | "
                          f"team%={point_data['team_fill']:.2f}")    

            # Progress update every 50 frames
            if idx % 50 == 0:
                print(f"  Progress: {idx}/{total_frames} frames processed")

    #Final summary
    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"  Total frames processed : {total_frames}")
    print(f"  Kill events detected   : {kill_count}")
    print(f"  Kill cam frames flagged: {killcam_count}")
    print(f"  Continuous data        → {continuous_path}")
    print(f"  Kill events            → {events_path}")

    if events_buffer:
        print(f"\nTop 5 turning points by weight:")
        top = sorted(
            events_buffer,
            key=lambda x: x["event_weight"],
            reverse=True
        )[:5]
        for i, e in enumerate(top, 1):
            print(f"  {i}. t={e['timestamp']}s | "
                  f"{e['killer_hero']} → {e['victim_hero']} | "
                  f"weight={e['event_weight']:+d} | "
                  f"type={e['kill_type']}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Marvel Rivals Game State Extractor")
    parser.add_argument("--mode", choices=["heuristic", "cnn"], default="heuristic", 
                        help="Choose 'heuristic' for template matching or 'cnn' for Deep Learning")
    args = parser.parse_args()
    print(f"--- STARTING PIPELINE IN {args.mode.upper()} MODE ---")
    run_pipeline(mode=args.mode)