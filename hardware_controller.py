import time
import threading
import queue

# Try gpiozero first (standard on Raspberry Pi OS Bookworm), then RPi.GPIO, or Mock mode on laptop
GPIO_BACKEND = "MOCK"
try:
    from gpiozero import OutputDevice
    GPIO_BACKEND = "GPIOZERO"
except ImportError:
    try:
        import RPi.GPIO as GPIO
        GPIO_BACKEND = "RPIGPIO"
    except ImportError:
        GPIO_BACKEND = "MOCK"


# --- PIN MAPPING (BCM Numbering) ---
# Change these numbers if you wired your 5 LEDs to different BCM GPIO pins!
BUTTON_PINS = {
    "LEFT": 17,        # Physical Pin 11
    "RIGHT": 27,       # Physical Pin 13
    "ROTATE_CW": 22,   # Physical Pin 15 (SNES 'A' or 'B')
    "ROTATE_CCW": 23,  # Physical Pin 16 (SNES 'Y' or 'B')
    "DROP": 24,        # Physical Pin 18 (SNES D-Pad Down)
}

# --- TIMING CONFIGURATION (Seconds) ---
# Mode 1: LED Placeholder Timings (long enough for human eyes to clearly see each blink!)
LED_MODE_TIMINGS = {
    "press_s": 0.12,       # LED stays ON for 120ms per tap
    "release_s": 0.08,     # 80ms gap between taps so consecutive LEFT -> LEFT blinks are distinct!
    "drop_hold_s": 0.35,   # Hold DROP LED longer so you can see the final lock-in!
}

# Mode 2: Real Solenoid Timings (fast mechanical strikes for SNES controller)
SOLENOID_MODE_TIMINGS = {
    "press_s": 0.035,      # 35ms strike time for TAU0730TM solenoid
    "release_s": 0.035,    # 35ms spring return recovery
    "drop_hold_s": 0.18,   # Soft-drop hold duration
}


class HardwareController:
    def __init__(self, use_solenoid_timings=False):
        self.timings = SOLENOID_MODE_TIMINGS if use_solenoid_timings else LED_MODE_TIMINGS
        self.cmd_queue = queue.Queue()
        self.running = True
        self.devices = {}

        if GPIO_BACKEND == "GPIOZERO":
            for name, pin in BUTTON_PINS.items():
                self.devices[name] = OutputDevice(pin, active_high=True, initial_value=False)
            print(f"💡 HardwareController online via gpiozero (BCM Pins: {BUTTON_PINS})")
        elif GPIO_BACKEND == "RPIGPIO":
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for name, pin in BUTTON_PINS.items():
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            print(f"💡 HardwareController online via RPi.GPIO (BCM Pins: {BUTTON_PINS})")
        else:
            print(f"💻 GPIO library not found — running in MOCK LED mode (BCM Pins: {BUTTON_PINS})")

        # Start non-blocking background worker thread so camera FPS never drops!
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _set_pin(self, action, state):
        if action not in BUTTON_PINS:
            return
        if GPIO_BACKEND == "GPIOZERO":
            if state:
                self.devices[action].on()
            else:
                self.devices[action].off()
        elif GPIO_BACKEND == "RPIGPIO":
            GPIO.output(BUTTON_PINS[action], GPIO.HIGH if state else GPIO.LOW)

    def dispatch_sequence(self, actions):
        """Queues a full list of actions (e.g. ['ROTATE_CW', 'LEFT', 'LEFT', 'DROP']) without blocking."""
        if not actions:
            return
        # Clear any stale unfinished commands if a new piece spawned
        while not self.cmd_queue.empty():
            try:
                self.cmd_queue.get_nowait()
                self.cmd_queue.task_done()
            except queue.Empty:
                break
        self.cmd_queue.put(list(actions))

    def _worker_loop(self):
        while self.running:
            try:
                sequence = self.cmd_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            for action in sequence:
                if not self.running:
                    break
                hold_time = self.timings["drop_hold_s"] if action == "DROP" else self.timings["press_s"]

                self._set_pin(action, True)
                time.sleep(hold_time)
                self._set_pin(action, False)
                time.sleep(self.timings["release_s"])

            self.cmd_queue.task_done()

    def cleanup(self):
        self.running = False
        for action in BUTTON_PINS:
            self._set_pin(action, False)
        if GPIO_BACKEND == "GPIOZERO":
            for dev in self.devices.values():
                dev.close()
        elif GPIO_BACKEND == "RPIGPIO":
            GPIO.cleanup()


# --- STANDALONE LED WIRING TESTER ---
# Run `python3 hardware_controller.py` on your Pi right now to test your 5 LEDs without the camera!
if __name__ == "__main__":
    print("\n🔧 Starting 5-LED Breadboard Wiring Test...")
    ctrl = HardwareController(use_solenoid_timings=False)
    try:
        test_moves = ["LEFT", "RIGHT", "ROTATE_CW", "ROTATE_CCW", "DROP"]
        for move in test_moves:
            print(f"  -> Blinking {move} (BCM GPIO {BUTTON_PINS[move]})")
            ctrl.dispatch_sequence([move])
            time.sleep(0.5)

        print("\n🎮 Simulating full AI Move: ROTATE_CW -> LEFT -> LEFT -> DROP")
        ctrl.dispatch_sequence(["ROTATE_CW", "LEFT", "LEFT", "DROP"])
        time.sleep(1.5)
        print("✅ LED Test Complete!")
    finally:
        ctrl.cleanup()