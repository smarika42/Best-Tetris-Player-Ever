import cv2
import json
import time

def main():
    print("📷 Initializing camera... please wait.")
    cap = cv2.VideoCapture(0) # 0 is usually the default camera. Change to 1 if you use an external USB webcam.
    
    # Warm up the camera (allows auto-exposure and auto-focus to settle)
    for _ in range(15):
        ret, frame = cap.read()
        time.sleep(0.1)
        
    ret, frame = cap.read()
    cap.release() # Release the camera so tetris_test.py can use it later!
    
    if not ret or frame is None:
        print("🚨 ERROR: Could not read from the camera. Is it plugged in and not used by another app?")
        return

    print("\n--- STEP 1: PLAYFIELD ---")
    print("Click and drag a box perfectly inside the borders of the main Tetris grid.")
    print("Press ENTER when done. (Press 'c' to cancel and redraw)")
    
    pf_box = cv2.selectROI("Calibration - Playfield", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Calibration - Playfield")

    print("\n--- STEP 2: NEXT PIECE BOX ---")
    print("Click and drag a box around the 'Next Piece' preview area.")
    print("Press ENTER when done.")
    
    next_box = cv2.selectROI("Calibration - Next Box", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Calibration - Next Box")

    # Save the exact pixel coordinates to the JSON file
    calibration_data = {
        "source": "camera",     # Hardcodes the source to camera!
        "camera_index": 0,
        "playfield": {
            "left": int(pf_box[0]),
            "top": int(pf_box[1]),
            "width": int(pf_box[2]),
            "height": int(pf_box[3])
        },
        "next_box": {
            "left": int(next_box[0]),
            "top": int(next_box[1]),
            "width": int(next_box[2]),
            "height": int(next_box[3])
        }
    }

    with open("calibration.json", "w") as f:
        json.dump(calibration_data, f, indent=4)

    print("\n✅ Camera calibration locked in and saved to calibration.json!")

if __name__ == "__main__":
    main()