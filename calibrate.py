import cv2
import json
import time

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False


class UniversalCamera:
    """Automatically uses Picamera2 (libcamera) for the Arducam OV5647 CSI ribbon camera,
    or falls back to standard OpenCV VideoCapture for USB webcams."""
    def __init__(self, index=0, width=640, height=480):
        self.use_picam = HAS_PICAMERA2
        self.picam2 = None
        self.cap = None

        if self.use_picam:
            try:
                print("🍓 Detected Picamera2! Initializing Arducam OV5647 CSI camera...")
                self.picam2 = Picamera2(camera_num=index)
                config = self.picam2.create_video_configuration(
                    main={"size": (width, height), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                return
            except Exception as e:
                print(f"⚠️ Picamera2 init failed ({e}), falling back to cv2.VideoCapture({index})...")
                self.use_picam = False

        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        if self.use_picam and self.picam2 is not None:
            frame = self.picam2.capture_array()
            if frame is None:
                return False, None
            # Note: In Picamera2, "RGB888" actually outputs BGR byte order in memory for OpenCV!
            return True, frame
        return self.cap.read()

    def release(self):
        if self.use_picam and self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()
        elif self.cap is not None:
            self.cap.release()


def main():
    print("📷 Initializing camera... please wait.")
    cam = UniversalCamera(index=0, width=640, height=480)

    # Warm up the sensor (allows auto-exposure and white balance to settle on the TV monitor)
    for _ in range(20):
        ret, frame = cam.read()
        time.sleep(0.05)

    if not ret or frame is None:
        print("🚨 ERROR: Could not read from the camera. Check the CSI ribbon cable and run 'rpicam-hello' to test!")
        cam.release()
        return

    print("\n--- LIVE AIMING PREVIEW ---")
    print("Position your Arducam OV5647 facing the 2-Player board.")
    print("Press SPACE or ENTER in the preview window to freeze the frame and draw boxes!")

    while True:
        ret, live_frame = cam.read()
        if ret and live_frame is not None:
            frame = live_frame.copy()
            cv2.putText(
                live_frame,
                "Aim camera, then press SPACE/ENTER to calibrate",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.imshow("Aim Camera (Press SPACE when ready)", live_frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32, ord('q')):  # ENTER or SPACE
            break

    cv2.destroyWindow("Aim Camera (Press SPACE when ready)")
    cam.release()

    print("\n--- STEP 1: PLAYFIELD ---")
    print("Click and drag a box perfectly inside the borders of the main Tetris grid.")
    print("Press ENTER or SPACE when done. (Press 'c' to cancel and redraw)")

    pf_box = cv2.selectROI("Calibration - Playfield", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Calibration - Playfield")

    print("\n--- STEP 2: NEXT PIECE BOX ---")
    print("Click and drag a box around the 'Next Piece' preview area.")
    print("Press ENTER or SPACE when done.")

    next_box = cv2.selectROI("Calibration - Next Box", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Calibration - Next Box")

    calibration_data = {
        "source": "camera",
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