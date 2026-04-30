import cv2
import numpy as np
import os



template_directory = "data/templates"

match_threshold = .64





def load_templates(templates_directory):
    """
    Loads hero images from the specified directory.
    Returns a dict mapping hero names to their images.

    -Converted to grayscale for NCC (reduces complexity)
    """
    templates = {}
    for filename in os.listdir(templates_directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            hero_name = filename.replace(".png", "").replace(".jpg", "")
            path = os.path.join(templates_directory, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                templates[hero_name] = img
                print(f"  Loaded template: {hero_name} {img.shape}")
    return templates


def normalized_cross_correlation(image_patch, template):
    """
    Computes the normalized cross-correlation between a template and an image.
    From scratch using numpy

    -- NCC = Σ[(T - mean(T)) * (I - mean(I))] / (std(T) * std(I) * n)

     Returns a score from -1.0 to 1.0
    - 1.0  = perfect match
    - 0.0  = no correlation
    - -1.0 = inverse match

    """

    #Resize to match template 

    if image_patch.shape != template.shape:
        image_patch = cv2.resize(image_patch, 
                                (template.shape[1], template.shape[0]))
    

    t = template.astype(np.float64)
    i = image_patch.astype(np.float64)

    #zero_mean patches
    t_zm = t - np.mean(t)
    i_zm = i - np.mean(i)


    numerator = np.sum(t_zm * i_zm)

    denominator = np.sqrt(np.sum(t_zm ** 2) * np.sum(i_zm ** 2))


    if denominator < 1e-10:
        return 0.0

    return numerator / denominator


def slide_and_match(kill_feed_gry, template, stride = 2):
    """
    Slide the template across the kill feed image and compute NCC
    at every position. Returns the best score and its location.

    Returns (best_score, (best_x, best_y)):
    - best_score: highest NCC score found (float)
    - best_loc: (x, y) position of best match
    - score_map: full 2D array of NCC scores for debugging
    """

    t_h, t_w = template.shape
    i_h, i_w = kill_feed_gry.shape

    #Score map dimensions
    output_h = (i_h - t_h) // stride + 1
    output_w = (i_w - t_w) // stride + 1

    score_map = np.zeros((output_h, output_w))
    best_score = -1.0
    best_loc = (0, 0)


    for row in range(output_h):
        for col in range(output_w):

            #Extract patch 
            y = row * stride 
            x = col * stride

            patch = kill_feed_gry[y:y+t_h, x:x+t_w]

            #Compute NCC
            score = normalized_cross_correlation(patch, template)
            score_map[row, col] = score

            if score > best_score:
                best_score = score
                best_loc = (x, y)

    return best_score, best_loc, score_map



def detect_heroes_in_killfeed(kill_feed_crop, templates, threshold=match_threshold):
    """
    Run template matching for all hero portraits against a kill feed crop.
    
    For each template, slides it across the kill feed and checks if
    the best NCC score exceeds our confidence threshold.
    
    Returns a list of detected heroes with their locations and scores:
    [
        {"hero": "psylocke", "score": 0.82, "location": (x, y)},
        {"hero": "jeff",     "score": 0.71, "location": (x, y)},
        ...
    ]
    """
    # Convert kill feed to grayscale for NCC
    kill_feed_gray = cv2.cvtColor(kill_feed_crop, cv2.COLOR_BGR2GRAY)

    detections = []

    for hero_name, template in templates.items():
        best_score, best_loc, score_map = slide_and_match(kill_feed_gray, template)

        if best_score < threshold:
            continue

        # Find ALL locations above threshold in score map
        t_h, t_w = template.shape
        locations = np.where(score_map >= threshold)

        for row, col in zip(locations[0], locations[1]):
            score = score_map[row, col]
            x = col * 2  # multiply by stride
            y = row * 2

            detections.append({
                "hero":     hero_name,
                "score":    round(float(score), 3),
                "location": (x, y),
                "t_shape":  (t_h, t_w),
            })

    # Sort by score descending so best matches are first
    detections = non_max_suppression(detections, overlap_thresh=0.3)
    detections.sort(key=lambda x: x["score"], reverse=True)
    return detections

def non_max_suppression(detections, overlap_thresh=0.3):
    """
    Remove duplicate detections that overlap too much.
    Keeps the highest scoring box when two boxes overlap significantly.
    """
    if not detections:
        return []

    # Convert to list of boxes (x1, y1, x2, y2, score)
    boxes = []
    for det in detections:
        x, y   = det["location"]
        t_h, t_w = det["t_shape"]
        boxes.append((x, y, x+t_w, y+t_h, det["score"], det))

    # Sort by score descending
    boxes.sort(key=lambda b: b[4], reverse=True)

    kept = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best[5])

        # Remove boxes that overlap too much with best
        remaining = []
        for box in boxes:
            
            ix1 = max(best[0], box[0])
            iy1 = max(best[1], box[1])
            ix2 = min(best[2], box[2])
            iy2 = min(best[3], box[3])

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            intersection = iw * ih

            # Compute union
            area_best = (best[2]-best[0]) * (best[3]-best[1])
            area_box  = (box[2]-box[0])   * (box[3]-box[1])
            union     = area_best + area_box - intersection

            iou = intersection / union if union > 0 else 0

            if iou < overlap_thresh:
                remaining.append(box)

        boxes = remaining

    return kept



def classify_kill_side(kill_feed_crop, location, template_shape):
    """
    Given a detected hero location, determine if they are the
    KILLER (left side) or VICTIM (right side) of the kill entry.
    
    We split the kill feed crop down the middle — left half = killer,
    right half = victim.
    
    Returns: "killer" or "victim"
    """
    frame_midpoint = kill_feed_crop.shape[1] // 2
    hero_x         = location[0]

    return "killer" if hero_x < frame_midpoint else "victim"


def draw_debug_matches(kill_feed_crop, detections, template_shapes):
    debug = kill_feed_crop.copy()
    for det in detections:
        x, y     = det["location"]
        hero     = det["hero"]
        score    = det["score"]
        t_h, t_w = det.get("t_shape", template_shapes.get(hero, (20, 20)))

        cv2.rectangle(debug, (x, y), (x+t_w, y+t_h), (0, 255, 0), 2)
        label = f"{hero} {score:.2f}"
        cv2.putText(debug, label, (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return debug


if __name__ == "__main__":
    print("Loading templates...")
    templates = load_templates(template_directory)
    print(f"Loaded {len(templates)} templates\n")

    # Test on a single kill feed frame
    test_path = "data/extracted_frames/kill_feed/t00096.51_f005488.png"
    frame     = cv2.imread(test_path)

    if frame is None:
        print(f"Could not load {test_path}")
    else:
        print(f"Running template matching on {test_path}...")
        detections = detect_heroes_in_killfeed(frame, templates)

        if not detections:
            print("No heroes detected above threshold")
        else:
            print(f"\nDetected {len(detections)} heroes:")
            for det in detections:
                side = classify_kill_side(frame, det["location"], 
                                         {det["hero"]: templates[det["hero"]].shape})
                print(f"  {det['hero']:25s} score={det['score']} side={side}")

        os.makedirs("results/debug_matches", exist_ok=True)
        template_shapes = {name: t.shape for name, t in templates.items()}
        debug_img = draw_debug_matches(frame, detections, template_shapes)
        cv2.imwrite("results/debug_matches/test_match.png", debug_img)
        print("\nSaved debug image to results/debug_matches/test_match.png")