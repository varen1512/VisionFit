import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer


# ---------------- MEDIA PIPE ----------------

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# ---------------- GLOBAL STATE ----------------

state = "IDLE"   # IDLE → RUNNING → PAUSED
counter = 0
stage = None
rep_lock = False


# ---------------- ANGLE FUNCTION ----------------

def angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = abs(radians * 180.0 / np.pi)

    if ang > 180:
        ang = 360 - ang

    return ang


# ---------------- VOICE (VOSK OFFLINE) ----------------

q = queue.Queue()

model = Model("vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(model, 16000)


def audio_callback(indata, frames, time, status):
    q.put(bytes(indata))


def voice_loop(on_command):
    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):

        print("🎤 Voice ready: say START / PAUSE / RESET")

        while True:
            data = q.get()

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")

                if text:
                    print("Heard:", text)

                    if "start" in text:
                        on_command("start")
                    elif "pause" in text:
                        on_command("pause")
                    elif "reset" in text:
                        on_command("reset")


# ---------------- COMMAND HANDLER ----------------

def handle_voice(cmd):
    global state, counter, stage, rep_lock

    if cmd == "start":
        state = "RUNNING"

    elif cmd == "pause":
        state = "PAUSED"

    elif cmd == "reset":
        counter = 0
        stage = None
        rep_lock = False


threading.Thread(target=voice_loop, args=(handle_voice,), daemon=True).start()


# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0)


with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape


        # ---------------- IDLE ----------------
        if state == "IDLE":
            cv2.putText(frame, "SAY 'START' TO BEGIN",
                        (90, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0, 255, 255), 3)

            cv2.putText(frame, "AI Gym Coach Ready",
                        (140, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

            cv2.imshow("AI Gym Coach", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            continue


        # ---------------- PAUSED ----------------
        if state == "PAUSED":
            cv2.putText(frame, "PAUSED", (230, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 4)

            cv2.putText(frame, f"TOTAL REPS: {counter}", (180, 260),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

            cv2.putText(frame, "Say START to resume",
                        (150, 320),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

            cv2.imshow("AI Gym Coach", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            continue


        # ---------------- RUNNING ----------------
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            lm = results.pose_landmarks.landmark

            shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            ang = angle(shoulder, elbow, wrist)

            # ---------------- REP LOGIC (FIXED) ----------------
            if ang > 160:
                stage = "up"
                rep_lock = False

            if ang < 90 and stage == "up" and not rep_lock:
                stage = "down"
                counter += 1
                rep_lock = True

            cv2.putText(frame, str(int(ang)),
                        tuple(np.multiply(elbow, [w, h]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

        except:
            pass


        # ---------------- UI ----------------
        cv2.rectangle(frame, (0, 0), (280, 120), (0, 0, 0), -1)

        cv2.putText(frame, f"REPS: {counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(frame, f"STATE: {state}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, "Voice: start / pause / reset",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)


        # ---------------- LANDMARKS ----------------
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )


        cv2.imshow("AI Gym Coach", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()