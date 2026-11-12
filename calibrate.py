import cv2
import json
import time

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False


class UniversalCamera:
    """Drives the Arducam IMX708 using the FULL 16:9 sensor area (no zoom crop!)
    with motorized autofocus + manual LensPosition support."""
    def __init__(self, index=0, width=640, height=360):
        self.use_picam = HAS_PICAMERA2
        self.picam2 = None
        self.cap = None
        self.lens_pos = 2.0  # Default ~0.5 meters (20 inches) if manual focus is used

        if self.use_picam:
            try:
                print("🍓 Initializing Arducam IMX708 (Full-Sensor Wide FOV + Autofocus)...")
                self.picam2 = Picamera2(camera_num=index)
                # Force the full 2304x1296 2x2 binned sensor mode so it NEVER crops/zooms in!
                config = self.picam2.create_video_configuration(
                    sensor={"output_size": (2304, 1296)},
                    main={"size": (width, height), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()

                # Enable Continuous Autofocus (AfMode = 2) and Fast AF Speed (AfSpeed = 1)
                try:
                    self.picam2.set_controls({"AfMode": 2, "AfSpeed": 1})
                    print("🔍 Continuous Autofocus enabled!")
                except Exception as af_err:
                    print(f"ℹ️ Autofocus control note: {af_err}")
                return
            except Exception as e:
                print(f"⚠️ Picamera2 init failed ({e}), falling back to cv2.VideoCapture({index})...")
                self.use_picam = False

        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def set_manual_focus(self, lens_pos):
        """Sets manual focus in diopters (0.0 = infinity, 2.0 = 50cm, 3.0 = 33cm, 4.0 = 25cm)."""
        self.lens_pos = max(0.0, min(10.0, float(lens_pos)))
        if self.use_picam and self.picam2 is not None:
            try:
                self.picam2.set_controls({"AfMode": 0, "LensPosition": self.lens_pos})
                print(f"🎯 Manual Focus set to LensPosition = {self.lens_pos:.2f} (~{100.0 / max(0.1, self.lens_pos):.0f} cm)")
            except Exception as e:
                print(f"⚠️ Could not set manual focus: {e}")

    def trigger_autofocus(self):
        if self.use_picam and self.picam2 is not None:
            try:
                self.picam2.set_controls({"AfMode": 2, "AfSpeed": 1})
                print("🔍 Switched back to Continuous Autofocus!")
            except Exception:
                pass

    def read(self):
        if self.use_picam and self.picam2 is not None:
            frame = self.picam2.capture_array()
            if frame is None:
                return False, None
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
    cam = UniversalCamera(index=0, width=640, height=360)

    for _ in range(25):
        ret, frame = cam.read()
        time.sleep(0.04)

    if not ret or frame is None:
        print("🚨 ERROR: Could not read from the camera.")
        cam.release()
        return

    print("\n--- LIVE AIMING & FOCUS PREVIEW ---")
    print("• Press '+' or '=' to focus CLOSER | Press '-' to focus FURTHER")
    print("• Press 'a' for Continuous Autofocus")
    print("• Press SPACE or ENTER when sharp to freeze and draw boxes!")

    manual_mode = False
    while True:
        ret, live_frame = cam.read()
        if ret and live_frame is not None:
            frame = live_frame.copy()
            focus_txt = f"Manual ({cam.lens_pos:.1f})" if manual_mode else "Auto (Press +/- to manual tune)"
            cv2.putText(
                live_frame,
                f"Focus: {focus_txt} | SPACE/ENTER to calibrate",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            cv2.imshow("Aim & Focus Camera (SPACE when ready)", live_frame)

        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32, ord('q')):  # ENTER or SPACE
            break
        elif key in (ord('+'), ord('=')):
            manual_mode = True
            cam.set_manual_focus(cam.lens_pos + 0.25)
        elif key == ord('-'):
            manual_mode = True
            cam.set_manual_focus(cam.lens_pos - 0.25)
        elif key == ord('a'):
            manual_mode = False
            cam.trigger_autofocus()

    cv2.destroyWindow("Aim & Focus Camera (SPACE when ready)")
    cam.release()

    print("\n--- STEP 1: PLAYFIELD ---")
    pf_box = cv2.selectROI("Calibration - Playfield", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Calibration - Playfield")

    print("\n--- STEP 2: NEXT PIECE BOX ---")
    next_box = cv2.selectROI("Calibration - Next Box", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Calibration - Next Box")

    calibration_data = {
        "source": "camera",
        "camera_index": 0,
        "lens_position": cam.lens_pos if manual_mode else None,
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

    print("\n✅ Camera calibration & focus saved to calibration.json!")


if __name__ == "__main__":
    main()