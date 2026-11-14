import os
import subprocess
import time

try:
    from gpiozero import OutputDevice
    USE_GPIOZERO = True
except ImportError:
    import RPi.GPIO as GPIO
    USE_GPIOZERO = False

# --- PIN DEFINITIONS (BCM) ---
BOOST_PWM_PIN = 18   # Physical Pin 12 (Hardware PWM0 -> Schematic 'Signal' / Q1)
SOLENOID_PIN = 17    # Physical Pin 11 (Digital Out   -> Schematic 'gpio' / R1 / Q2)

# --- EXACT ARDUINO TIMER1 EQUIVALENT ---
# Arduino: 16MHz / (511 + 1) = 31,250 Hz -> Period = 32,000 nanoseconds
# Arduino: 77 / 512 = 15.04% duty cycle  -> Duty_cycle = 4,812 nanoseconds
PWM_PERIOD_NS = 32000
PWM_DUTY_NS = 4812     # 15% duty cycle (NEVER exceed 25% on an open-loop boost converter!)


class HardwareBoostPWM:
    """Drives BCM GPIO 18 using the Linux kernel hardware PWM peripheral at 31.25 kHz."""
    def __init__(self, period_ns=PWM_PERIOD_NS, duty_ns=PWM_DUTY_NS):
        self.pwm_dir = None
        self.period_ns = period_ns
        self.duty_ns = duty_ns
        self._setup_hardware_pwm()

    def _setup_hardware_pwm(self):
        # 1. Ensure the hardware pwm overlay is loaded on GPIO 18
        if not os.path.exists("/sys/class/pwm/pwmchip0"):
            print("🔧 Loading hardware PWM overlay on BCM GPIO 18...")
            subprocess.run(["sudo", "dtoverlay", "pwm", "pin=18", "func=2"], check=False)
            time.sleep(0.3)

        # Find the active pwmchip directory
        chips = sorted(os.listdir("/sys/class/pwm")) if os.path.exists("/sys/class/pwm") else []
        if not chips:
            raise RuntimeError(
                "No /sys/class/pwm/pwmchip* found! Run: 'sudo dtoverlay pwm pin=18 func=2' in terminal."
            )

        chip_path = f"/sys/class/pwm/{chips[0]}"
        self.pwm_dir = f"{chip_path}/pwm0"

        if not os.path.exists(self.pwm_dir):
            self._write_sysfs(f"{chip_path}/export", "0")
            time.sleep(0.1)

        # Disable first before changing period/duty
        self._write_sysfs(f"{self.pwm_dir}/enable", "0")
        self._write_sysfs(f"{self.pwm_dir}/period", str(self.period_ns))
        self._write_sysfs(f"{self.pwm_dir}/duty_cycle", str(self.duty_ns))
        self._write_sysfs(f"{self.pwm_dir}/enable", "1")
        print(f"⚡ Boost Converter Hardware PWM ONLINE on GPIO 18 (31.25 kHz, 15.0% duty cycle)!")

    def _write_sysfs(self, path, value):
        try:
            with open(path, "w") as f:
                f.write(value)
        except PermissionError:
            subprocess.run(["sudo", "sh", "-c", f"echo {value} > {path}"], check=True)

    def stop(self):
        if self.pwm_dir and os.path.exists(self.pwm_dir):
            try:
                self._write_sysfs(f"{self.pwm_dir}/duty_cycle", "0")
                self._write_sysfs(f"{self.pwm_dir}/enable", "0")
                print("🛑 Boost Converter Hardware PWM safely turned OFF.")
            except Exception as e:
                print(f"⚠️ Error stopping PWM: {e}")


def set_solenoid(sol_dev, state):
    if USE_GPIOZERO:
        if state:
            sol_dev.on()
        else:
            sol_dev.off()
    else:
        GPIO.output(SOLENOID_PIN, GPIO.HIGH if state else GPIO.LOW)


def main():
    if USE_GPIOZERO:
        solenoid = OutputDevice(SOLENOID_PIN, active_high=True, initial_value=False)
    else:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(SOLENOID_PIN, GPIO.OUT, initial=GPIO.LOW)
        solenoid = None

    boost_pwm = None
    try:
        boost_pwm = HardwareBoostPWM()
        # Give C2 capacitor 0.5 seconds to charge up from the 31.25kHz boost converter!
        print("🔋 Charging boost capacitor C2 for 0.5s...")
        time.sleep(0.5)

        print("\n========================================================")
        print("🍓 RASPBERRY PI BOOST + SOLENOID TESTER")
        print("   Boost Pin   : BCM GPIO 18 (Physical Pin 12) @ 31.25kHz")
        print("   Solenoid Pin: BCM GPIO 17 (Physical Pin 11)")
        print("========================================================")
        print("Commands:")
        print("  [ENTER] -> Fire single 60ms button tap (recharges between taps)")
        print("  loop    -> Run your friend's exact 1s ON / 1s OFF Arduino loop")
        print("  q       -> Quit & shut down Boost PWM + Solenoid safely")
        print("========================================================\n")

        while True:
            cmd = input("Press ENTER to fire tap (or 'loop' / 'q'): ").strip().lower()
            if cmd == 'q':
                break
            elif cmd == 'loop':
                print("🔄 Running Arduino 1s LOW / 1s HIGH loop (Press Ctrl+C to stop)...")
                while True:
                    set_solenoid(solenoid, False)
                    time.sleep(1.0)
                    print("  -> Solenoid HIGH")
                    set_solenoid(solenoid, True)
                    time.sleep(1.0)
                    print("  -> Solenoid LOW")
            else:
                print("⚡ Firing Solenoid (60ms strike)!")
                set_solenoid(solenoid, True)
                time.sleep(0.060)
                set_solenoid(solenoid, False)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user! Shutting down safely...")
    finally:
        set_solenoid(solenoid, False)
        if boost_pwm is not None:
            boost_pwm.stop()
        if USE_GPIOZERO and solenoid is not None:
            solenoid.close()
        elif not USE_GPIOZERO:
            GPIO.cleanup()
        print("✅ All pins pulled LOW.")


if __name__ == "__main__":
    main()