# ============================================================================
# redRover — Multi-Modal Facility Health Robot
# Sphero RVR+ Patrol Script for Sphero EDU (Mac App)
# ============================================================================
#
# PURPOSE:
#   This program turns a Sphero RVR+ into a factory-floor patrol robot.
#   It drives a rectangular route, stopping at 4 "machine stations" to
#   measure vibration using the on-board IMU (accelerometer + gyroscope).
#   Each station gets a health diagnosis: GREEN (healthy), YELLOW (warning),
#   or RED (critical) based on vibration thresholds.
#
# HOW IT WORKS:
#   1. Boot-up LED sequence + voice announcement
#   2. Drive to each of 4 stations in a rectangular pattern
#   3. At each station: collect 20 IMU samples over ~2 seconds
#   4. Compute RMS acceleration, peak acceleration, gyro magnitude
#   5. Classify health and announce diagnosis via text-to-speech
#   6. Return home after all stations are visited
#   7. Rainbow celebration + summary announcement
#
# USAGE:
#   Copy-paste this entire file into the Sphero EDU custom program editor.
#   No imports needed — all functions are pre-loaded in the EDU sandbox.
#
# HACKATHON: redRover — Multi-Modal Facility Health Robot
# ============================================================================


async def start_program():

    # -----------------------------------------------------------------
    # CONFIGURATION
    # -----------------------------------------------------------------

    # Patrol speed (0-255). Moderate speed for indoor demo.
    PATROL_SPEED = 80

    # How long to drive between stations (seconds). Tune for your room size.
    # At speed 80, ~2.5 seconds covers roughly 1-1.5 meters.
    DRIVE_DURATION = 2.5

    # IMU sampling: how many readings per station and delay between them
    NUM_SAMPLES = 20
    SAMPLE_INTERVAL = 0.1  # seconds between samples (20 samples ~ 2 sec total)

    # Vibration health thresholds (in g-force for acceleration)
    HEALTHY_RMS_THRESHOLD = 0.15
    WARNING_PEAK_THRESHOLD = 0.5

    # Station definitions: name, heading to reach it, label for TTS
    # We patrol in a rectangle: forward, right, backward, left
    STATIONS = [
        {"name": "Station Alpha",   "heading": 0,   "label": "Alpha"},
        {"name": "Station Bravo",   "heading": 90,  "label": "Bravo"},
        {"name": "Station Charlie", "heading": 180, "label": "Charlie"},
        {"name": "Station Delta",   "heading": 270, "label": "Delta"},
    ]

    # Colors as dicts (Sphero EDU format: {'r': R, 'g': G, 'b': B})
    COLOR_OFF     = {'r': 0,   'g': 0,   'b': 0}
    COLOR_GREEN   = {'r': 0,   'g': 255, 'b': 0}
    COLOR_BLUE    = {'r': 0,   'g': 0,   'b': 255}
    COLOR_CYAN    = {'r': 0,   'g': 255, 'b': 255}
    COLOR_YELLOW  = {'r': 255, 'g': 255, 'b': 0}
    COLOR_RED     = {'r': 255, 'g': 0,   'b': 0}
    COLOR_WHITE   = {'r': 255, 'g': 255, 'b': 255}
    COLOR_MAGENTA = {'r': 255, 'g': 0,   'b': 255}

    # -----------------------------------------------------------------
    # HELPER FUNCTIONS
    # -----------------------------------------------------------------

    def set_all_leds(color):
        """Set main, front, and back LEDs to the same color."""
        set_main_led(color)
        set_front_led(color)
        set_back_led(color)

    async def flash_led(color, times, on_time, off_time):
        """Flash all LEDs a given color a set number of times."""
        for i in range(times):
            set_all_leds(color)
            await delay(on_time)
            set_all_leds(COLOR_OFF)
            await delay(off_time)

    async def rainbow_cycle(cycles, step_delay):
        """Cycle through rainbow colors on all LEDs."""
        rainbow = [
            {'r': 255, 'g': 0,   'b': 0},    # Red
            {'r': 255, 'g': 127, 'b': 0},    # Orange
            {'r': 255, 'g': 255, 'b': 0},    # Yellow
            {'r': 0,   'g': 255, 'b': 0},    # Green
            {'r': 0,   'g': 0,   'b': 255},  # Blue
            {'r': 75,  'g': 0,   'b': 130},  # Indigo
            {'r': 148, 'g': 0,   'b': 211},  # Violet
        ]
        for cycle in range(cycles):
            for color in rainbow:
                set_all_leds(color)
                await delay(step_delay)

    async def collect_imu_samples(num_samples, interval):
        """
        Collect acceleration and gyroscope data from the IMU.
        Returns a dict with lists of accel magnitudes and gyro magnitudes.
        """
        accel_magnitudes = []
        gyro_magnitudes = []

        for i in range(num_samples):
            # Read raw accelerometer (returns x, y, z in g-force)
            accel = await get_acceleration()
            ax = accel["x"]
            ay = accel["y"]
            az = accel["z"]

            # Magnitude of acceleration vector (subtract 1g for gravity)
            accel_mag = ((ax * ax) + (ay * ay) + (az * az)) ** 0.5
            vibration = abs(accel_mag - 1.0)
            accel_magnitudes.append(vibration)

            # Read gyroscope (returns x, y, z in degrees/sec)
            gyro = await get_gyroscope()
            gx = gyro["x"]
            gy = gyro["y"]
            gz = gyro["z"]

            gyro_mag = ((gx * gx) + (gy * gy) + (gz * gz)) ** 0.5
            gyro_magnitudes.append(gyro_mag)

            await delay(interval)

        return {
            "accel": accel_magnitudes,
            "gyro": gyro_magnitudes,
        }

    def analyze_vibration(samples):
        """
        Compute vibration health metrics from collected IMU samples.
        Returns a dict with: rms_accel, peak_accel, avg_gyro, health, color.
        """
        accel_data = samples["accel"]
        gyro_data = samples["gyro"]
        n = len(accel_data)

        # RMS Acceleration — overall vibration energy
        sum_sq = 0.0
        for val in accel_data:
            sum_sq = sum_sq + (val * val)
        rms_accel = (sum_sq / n) ** 0.5

        # Peak Acceleration — maximum single vibration spike
        peak_accel = 0.0
        for val in accel_data:
            if val > peak_accel:
                peak_accel = val

        # Average Gyroscope Magnitude — rotational vibration
        gyro_sum = 0.0
        for val in gyro_data:
            gyro_sum = gyro_sum + val
        avg_gyro = gyro_sum / n

        # Health Classification
        if peak_accel > WARNING_PEAK_THRESHOLD:
            health = "CRITICAL"
            color = COLOR_RED
        elif rms_accel > HEALTHY_RMS_THRESHOLD:
            health = "WARNING"
            color = COLOR_YELLOW
        else:
            health = "HEALTHY"
            color = COLOR_GREEN

        return {
            "rms_accel": rms_accel,
            "peak_accel": peak_accel,
            "avg_gyro": avg_gyro,
            "health": health,
            "color": color,
        }

    def build_diagnosis_message(station_label, analysis):
        """Build a human-readable TTS diagnosis string."""
        health = analysis["health"]
        if health == "HEALTHY":
            return station_label + " is healthy. Vibration nominal."
        elif health == "WARNING":
            return station_label + " warning. Elevated vibration detected."
        else:
            return station_label + " critical! High vibration alert!"

    # -----------------------------------------------------------------
    # BOOT-UP SEQUENCE
    # -----------------------------------------------------------------

    # Reset heading so "forward" is wherever the robot is currently facing
    await reset_aim()

    set_all_leds(COLOR_OFF)
    await delay(0.3)

    # Dramatic power-on LED sequence
    await flash_led(COLOR_GREEN, 3, 0.2, 0.15)

    # Solid green = system ready
    set_all_leds(COLOR_GREEN)
    await delay(0.5)

    # Quick white flash = sensors online
    await flash_led(COLOR_WHITE, 2, 0.1, 0.1)
    set_all_leds(COLOR_GREEN)
    await delay(0.3)

    # Voice announcement
    await speak("red Rover patrol starting. Initiating facility health scan.")
    await delay(2.0)

    # -----------------------------------------------------------------
    # MAIN PATROL LOOP
    # -----------------------------------------------------------------

    healthy_count = 0
    station_results = []

    for station_index in range(len(STATIONS)):
        station = STATIONS[station_index]
        station_name = station["name"]
        station_label = station["label"]
        station_heading = station["heading"]

        # --- NAVIGATE TO STATION ---
        await speak("Navigating to " + station_name)
        await delay(1.0)

        # Drive to station with blue LEDs
        set_all_leds(COLOR_BLUE)
        await roll(station_heading, PATROL_SPEED, DRIVE_DURATION)
        await stop_roll()
        await delay(0.5)

        # --- ARRIVE AND MEASURE ---
        set_all_leds(COLOR_CYAN)
        await speak("Arrived at " + station_name + ". Beginning vibration analysis.")
        await delay(1.5)

        # Pulsing cyan effect while "calibrating sensors"
        for pulse in range(3):
            set_main_led(COLOR_CYAN)
            set_front_led(COLOR_WHITE)
            await delay(0.2)
            set_front_led(COLOR_CYAN)
            set_main_led(COLOR_WHITE)
            await delay(0.2)
        set_all_leds(COLOR_CYAN)

        # --- COLLECT IMU DATA ---
        samples = await collect_imu_samples(NUM_SAMPLES, SAMPLE_INTERVAL)

        # --- ANALYZE VIBRATION ---
        analysis = analyze_vibration(samples)

        # --- DISPLAY AND ANNOUNCE RESULTS ---
        set_all_leds(analysis["color"])
        diagnosis = build_diagnosis_message(station_label, analysis)
        await speak(diagnosis)
        await delay(2.0)

        # Track results
        if analysis["health"] == "HEALTHY":
            healthy_count = healthy_count + 1
        station_results.append({
            "label": station_label,
            "health": analysis["health"],
        })

        # Brief status hold so audience can see the LED color
        await delay(1.0)
        await flash_led(analysis["color"], 2, 0.15, 0.1)

    # -----------------------------------------------------------------
    # RETURN HOME
    # -----------------------------------------------------------------

    await speak("All stations scanned. Returning to base.")
    await delay(1.5)

    # Rectangle patrol brings us back near start — short docking nudge
    set_all_leds(COLOR_MAGENTA)
    await roll(0, 40, 1.0)
    await stop_roll()
    await delay(0.5)

    # -----------------------------------------------------------------
    # PATROL SUMMARY AND CELEBRATION
    # -----------------------------------------------------------------

    summary = "Patrol complete. " + str(healthy_count) + " of 4 stations healthy."
    await speak(summary)
    await delay(2.0)

    # Call out any non-healthy stations
    for result in station_results:
        if result["health"] != "HEALTHY":
            await speak(result["label"] + " requires maintenance attention. Status: " + result["health"])
            await delay(1.5)

    # Rainbow celebration
    await rainbow_cycle(3, 0.15)

    # Final status LED
    has_critical = False
    has_warning = False
    for result in station_results:
        if result["health"] == "CRITICAL":
            has_critical = True
        if result["health"] == "WARNING":
            has_warning = True

    if has_critical:
        set_all_leds(COLOR_RED)
        await speak("Alert: Critical vibration detected. Maintenance required.")
    elif has_warning:
        set_all_leds(COLOR_YELLOW)
        await speak("Caution: Elevated vibration at some stations. Monitor closely.")
    else:
        set_all_leds(COLOR_GREEN)
        await speak("All clear. Facility vibration levels nominal. red Rover standing by.")

    await delay(3.0)
    set_all_leds(COLOR_OFF)

    return


# ============================================================================
# END OF PATROL SCRIPT
# ============================================================================
