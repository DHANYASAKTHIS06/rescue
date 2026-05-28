import joblib
import pandas as pd
import numpy as np
import cv2
import base64
import time
import threading
import warnings
import sys

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Bulletproof Headless MediaPipe Resolution ───────────────────────────────
import mediapipe as mp

try:
    # Standard cloud/headless path
    from mediapipe.solutions import hands as mp_hands
    from mediapipe.solutions import drawing_utils as mp_drawing
except ImportError:
    # Alternative local fallback path
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing

# Suppress scikit-learn version mismatch warnings in the Render logs
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Load ML models ─────────────────────────────────────────────────────────────
try:
    model         = joblib.load("knn_regressor_model.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
except Exception as e:
    print(f"CRITICAL Warning during model loading: {e}")
    model = None
    label_encoder = None

app = Flask(__name__)
CORS(app)

# ── MediaPipe Setup ───────────────────────────────────────────────────────────
try:
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5, # Slightly lowered to catch moving hands easier
        min_tracking_confidence=0.5
    )
except Exception as e:
    print(f"MediaPipe initialization warning: {e}")
    hands = None

# ── Wave Detector ──────────────────────────────────────────────────────────────
class WaveDetector:
    # Optimized parameters to handle slower web/network frame rates over HTTP
    HISTORY_SECONDS = 3.0  
    MIN_CROSSINGS   = 2    
    CENTRE_DEADBAND = 0.06 

    def __init__(self):
        self.positions = []   # (timestamp, norm_x)

    def update(self, norm_x: float) -> str:
        now = time.time()
        self.positions.append((now, norm_x))
        cutoff = now - self.HISTORY_SECONDS
        self.positions = [(t, x) for t, x in self.positions if t >= cutoff]

        if len(self.positions) < 3:
            return "IDLE"

        xs     = [x for _, x in self.positions]
        centre = 0.5
        above  = [x > centre + self.CENTRE_DEADBAND for x in xs]
        below  = [x < centre - self.CENTRE_DEADBAND for x in xs]

        crossings = sum(
            1 for i in range(1, len(xs))
            if (above[i] != above[i - 1]) or (below[i] != below[i - 1])
        )

        if crossings >= self.MIN_CROSSINGS:
            recent = [x for x in xs[-6:] if abs(x - centre) > self.CENTRE_DEADBAND]
            if len(recent) >= 2:
                return "WAVE_RIGHT" if recent[-1] > recent[0] else "WAVE_LEFT"
            return "WAVE_LEFT"

        return "IDLE"


wave_detector   = WaveDetector()
_cooldown_until = 0.0
_state_lock      = threading.Lock()

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Rescue AI Gesture Detection API Running"})


@app.route("/process-frame", methods=["POST"])
def process_frame():
    """
    Receives a base64-encoded JPEG frame from the browser webcam,
    runs MediaPipe hand detection + wave logic,
    returns gesture state + annotated frame (base64 JPEG).
    """
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

        # Mirror frame for intuitive orientation alignment
        frame = cv2.flip(frame, 1)

        gesture_label = "IDLE"
        is_emergency  = False
        predicted_gesture = "IDLE"
        encoded_prediction = 0

        if hands is not None:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    try:
                        mp_drawing.draw_landmarks(
                            frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0),    thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                        )
                    except Exception:
                        pass 

                    # Extract wrist horizontal tracking metrics
                    wrist_x  = hand_lm.landmark[0].x
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

                # KNN calculation via landmarks (Isolated to prevent variable overwrites)
                if model is not None and label_encoder is not None:
                    lm              = results.multi_hand_landmarks[0].landmark
                    feature_indices = [0, 4, 8, 12, 16, 20]
                    features        = [lm[i].x for i in feature_indices]
                    input_df        = pd.DataFrame([features], columns=["ax","ay","az","gx","gy","gz"])
                    prediction      = model.predict(input_df)
                    encoded_prediction = int(round(prediction[0]))
                    encoded_prediction = max(0, min(encoded_prediction, len(label_encoder.classes_) - 1))
                    predicted_gesture  = label_encoder.inverse_transform([encoded_prediction])[0]

        # Annotate processed output frame
        colour = (0, 0, 255) if is_emergency else (0, 255, 0)
        label  = "EMERGENCY!" if is_emergency else gesture_label
        cv2.putText(frame, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        out_b64   = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "success":            True,
            "gesture":            gesture_label,      # Heuristic wave status
            "emergency":          is_emergency,
            "predicted_gesture":  str(predicted_gesture), # ML static prediction status
            "encoded_prediction": int(encoded_prediction),
            "annotated_frame":    out_b64
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Legacy endpoint — manual sensor input still supported."""
    if model is None or label_encoder is None:
        return jsonify({"success": False, "error": "ML Model failed to load on server boot."}), 500

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
        predicted_gesture  = label_encoder.inverse_transform([encoded_prediction])[0]

        return jsonify({
            "success":            True,
            "predicted_gesture":  str(predicted_gesture),
            "encoded_prediction": int(encoded_prediction)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
