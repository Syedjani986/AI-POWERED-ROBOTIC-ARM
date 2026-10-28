
import time
import cv2
import numpy as np
import speech_recognition as sr
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

SENSOR_PATHS = [
    "/NiryoOne/visionSensor",
    "/visionSensor",
]

COLOR_RANGES = {
    "red": [
        ((0, 100, 70), (10, 255, 255)),
        ((170, 100, 70), (180, 255, 255)),
    ],
    "yellow": [((18, 80, 80), (38, 255, 255))],
    "blue": [((90, 70, 50), (135, 255, 255))],
}


def find_sensor(sim):
    for path in SENSOR_PATHS:
        try:
            return sim.getObject(path)
        except Exception:
            pass
    raise RuntimeError(
        "Vision sensor not found. Check its exact path in CoppeliaSim."
    )


def read_frame(sim, sensor):
    buffer, resolution = sim.getVisionSensorImg(sensor)
    width, height = int(resolution[0]), int(resolution[1])
    image = np.frombuffer(buffer, dtype=np.uint8)

    if image.size != width * height * 3:
        raise RuntimeError("Unexpected vision sensor image format.")

    image = image.reshape(height, width, 3)
    image = cv2.flip(image, 0)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def detect_color(frame, color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in COLOR_RANGES[color]:
        mask |= cv2.inRange(
            hsv, np.array(lower, np.uint8), np.array(upper, np.uint8)
        )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = [
        c for c in contours if cv2.contourArea(c) >= 30
    ]

    if not candidates:
        return None, mask

    contour = max(candidates, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, w, h), mask


def main():
    client = RemoteAPIClient()
    sim = client.require("sim")
    sensor = find_sensor(sim)

    recognizer = sr.Recognizer()
    print("Connected. Start the simulation in CoppeliaSim.")
    print("Say: red cube, yellow cube, or blue cube.")

    last_step = 0
    found = False

    with sr.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=1)

        while True:
            try:
                print("\nListening...")
                audio = recognizer.listen(
                    mic, timeout=6, phrase_time_limit=5
                )
                spoken = recognizer.recognize_google(
                    audio, language="en-IN"
                ).lower()
                print("Heard:", spoken)

                if "exit" in spoken or "quit" in spoken:
                    break

                color = next(
                    (c for c in COLOR_RANGES if c in spoken), None
                )
                if color is None:
                    print("Say red, yellow, or blue cube.")
                    continue

                # Reset the handshake before starting a new scan.
                sim.clearInt32Signal("scan_step")
                sim.clearInt32Signal("scan_ack")
                sim.clearInt32Signal("object_found")
                sim.setStringSignal("voice_command", color)

                last_step = 0
                found = False
                deadline = time.time() + 90

                while time.time() < deadline and not found:
                    step = sim.getInt32Signal("scan_step")

                    if step is not None and step > last_step:
                        # Let the arm and camera settle briefly.
                        time.sleep(0.2)
                        frame = read_frame(sim, sensor)
                        bbox, mask = detect_color(frame, color)

                        if bbox is not None:
                            x, y, w, h = bbox
                            cv2.rectangle(
                                frame, (x, y), (x+w, y+h),
                                (0, 255, 0), 2
                            )
                            print(
                                f"{color} detected in camera image "
                                f"at pixel ({x+w//2}, {y+h//2})"
                            )
                            found = True
                            sim.setInt32Signal("object_found", 1)
                        else:
                            sim.setInt32Signal("object_found", 0)
                            print(f"Scan step {step}: not detected")

                        cv2.imshow("Wrist Camera", frame)
                        cv2.imshow("Color Mask", mask)
                        cv2.waitKey(1)

                        sim.setInt32Signal("scan_ack", step)
                        last_step = step

                    time.sleep(0.05)

                if not found:
                    print("No matching color detected during scan.")

                print(
                    "Vision scan finished. Automatic grasping is not "
                    "enabled by this scanning version."
                )

            except sr.WaitTimeoutError:
                print("No speech detected.")
            except sr.UnknownValueError:
                print("Could not understand. Try again.")
            except sr.RequestError as exc:
                print("Speech recognition error:", exc)
            except Exception as exc:
                print("Error:", exc)
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
