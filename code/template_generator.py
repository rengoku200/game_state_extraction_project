import cv2
import os

def main():
    # The exact size your pipeline needs for the math to work
    TARGET_WIDTH = 50
    TARGET_HEIGHT = 24
    
    test_frame_path = "/Users/baidymbaye/cs1430/game_state_extraction_project/data/extracted_frames/kill_feed/t00148.21_f008428.png"
    
    img = cv2.imread(test_frame_path)
    if img is None:
        print(f"Error: Could not load image at {test_frame_path}")
        return

    print("========================================")
    print(f" GOAL: We need exactly {TARGET_WIDTH}x{TARGET_HEIGHT} pixels.")
    print(" 1. Draw a box tightly around a hero's face.")
    print(" 2. Don't worry about being perfectly accurate.")
    print(" 3. Python will automatically fix the size for you.")
    print("========================================\n")

    roi = cv2.selectROI("Template Generator", img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = roi

    if w > 0 and h > 0:
        # 1. Crop whatever messy box you drew
        cropped_face = img[y:y+h, x:x+w]
        
        # 2. THE MAGIC FIX: Mathematically force it to exactly 50x24
        standardized_face = cv2.resize(cropped_face, (TARGET_WIDTH, TARGET_HEIGHT))
        
        hero_name = input("What hero is this? (e.g., 'psylocke'): ")
        
        os.makedirs("data/templates", exist_ok=True)
        save_path = f"data/templates/{hero_name}.png"
        
        # 3. Save the perfect version, not the messy version
        cv2.imwrite(save_path, standardized_face)
        
        print(f"\nSUCCESS! You drew {w}x{h}, but we saved a perfect {TARGET_WIDTH}x{TARGET_HEIGHT} template!")
    else:
        print("\nNo box selected. Exiting.")

if __name__ == "__main__":
    main()