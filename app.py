import joblib
import pandas as pd
import numpy as np
import cv2
import base64
import time
import threading

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── New mediapipe API (0.10+) ──────────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image, ImageFormat

# ── Load ML models ─────────────────────────────────────────────────────────────
model         = joblib.load("knn_regressor_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

app = Flask(__name__)
CORS(app)

# ── Hand Landmarker setup (new API) ────────────────────────────────────────────
import urllib.request, os

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Downloaded.")

base_options    = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
hand_options    = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

# ── Wave Detector ──────────────────────────────────────────────────────────────
class WaveDetector:
    HISTORY_SECONDS = 1.5
    MIN_CROSSINGS   = 3
    CENTRE_DEADBAND = 0.08

    def __init__(self):
        self.positions = []

    def update(self, norm_x: float) -> str:
        now = time.time()
        self.positions.append((now, norm_x))
        cutoff = now - self.HISTORY_SECONDS
        self.positions = [(t, x) for t, x in self.positions if t >= cutoff]

        if len(self.positions) < 4:
            return "IDLE"

        xs     = [x for _, x in self.positions]
        centre = 0.5
        above  = [x > centre + self.CENTRE_DEADBAND for x in xs]
        below  = [x < centre - self.CENTRE_DEADBAND for x in xs]

        crossings = sum(
            1 for i in range(1, len(xs))
            if (above[i] != above[i-1]) or (below[i] != below[i-1])
        )

        if crossings >= self.MIN_CROSSINGS:
            recent = [x for x in xs[-6:] if abs(x - centre) > self.CENTRE_DEADBAND]
            if len(recent) >= 2:
                return "WAVE_RIGHT" if recent[-1] > recent[0] else "WAVE_LEFT"
            return "WAVE_LEFT"

        return "IDLE"


wave_detector   = WaveDetector()
_cooldown_until = 0.0
_state_lock     = threading.Lock()


# ── Helper: draw landmarks manually ───────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def draw_landmarks(frame, landmarks, h, w):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Rescue AI Gesture Detection API Running"})


@app.route("/process-frame", methods=["POST"])
def process_frame():
    global _cooldown_until

    try:
        data      = request.get_json(force=True)
        b64_frame = data.get("frame", "")

        if not b64_frame:
            return jsonify({"success": False, "error": "No frame provided"}), 400

        if "," in b64_frame:
            b64_frame = b64_frame.split(",", 1)[1]

        img_bytes = base64.b64decode(b64_frame)
        np_arr    = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "Could not decode frame"}), 400

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Convert BGR -> RGB for mediapipe
        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image   = Image(image_format=ImageFormat.SRGB, data=rgb)
        result     = hand_landmarker.detect(mp_image)

        gesture_label = "IDLE"
        is_emergency  = False

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, landmarks, h, w)

            wrist_x  = landmarks[0].x
            detected = wave_detector.update(wrist_x)

            if detected in ("WAVE_LEFT", "WAVE_RIGHT"):
                now = time.time()
                with _state_lock:
                    if now > _cooldown_until:
                        gesture_label   = detected
                        is_emergency    = True
                        _cooldown_until = now + 3.0
                    else:
                        gesture_label = detected
                        is_emergency  = False
            else:
                gesture_label = "IDLE"

        # Annotate frame
        colour = (0, 0, 255) if is_emergency else (0, 255, 0)
        label  = "EMERGENCY!" if is_emergency else gesture_label
        cv2.putText(frame, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        out_b64   = base64.b64encode(buffer).decode("utf-8")

        # KNN prediction
        predicted_gesture  = gesture_label
        encoded_prediction = 0

        if result.hand_landmarks:
            lm              = result.hand_landmarks[0]
            feature_indices = [0, 4, 8, 12, 16, 20]
            features        = [lm[i].x for i in feature_indices]
            input_df        = pd.DataFrame([features], columns=["ax","ay","az","gx","gy","gz"])
            prediction      = model.predict(input_df)
            encoded_prediction = int(round(prediction[0]))
            encoded_prediction = max(0, min(encoded_prediction, len(label_encoder.classes_) - 1))
            predicted_gesture  = str(label_encoder.inverse_transform([encoded_prediction])[0])

        return jsonify({
            "success":            True,
            "gesture":            gesture_label,
            "emergency":          is_emergency,
            "predicted_gesture":  predicted_gesture,
            "encoded_prediction": encoded_prediction,
            "annotated_frame":    out_b64
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data     = request.get_json()
        required = ["ax", "ay", "az", "gx", "gy", "gz"]
        missing  = [f for f in required if f not in data]

        if missing:
            return jsonify({"success": False, "error": f"Missing: {missing}"}), 400

        input_df = pd.DataFrame(
            [[float(data[f]) for f in required]],
            columns=required
        )
        prediction         = model.predict(input_df)
        encoded_prediction = int(round(prediction[0]))
        encoded_prediction = max(0, min(encoded_prediction, len(label_encoder.classes_) - 1))
        predicted_gesture  = str(label_encoder.inverse_transform([encoded_prediction])[0])

        return jsonify({
            "success":            True,
            "predicted_gesture":  predicted_gesture,
            "encoded_prediction": encoded_prediction
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
