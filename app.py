import joblib
import pandas as pd
import numpy as np
import cv2
import base64
import time
import threading
import warnings
import os
import io
import tempfile

warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
from flask_cors import CORS

import speech_recognition as sr

# ── ML Models ──────────────────────────────────────────────────────────────────
try:
    model         = joblib.load("knn_regressor_model.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
except Exception as e:
    print(f"Model loading warning: {e}")
    model         = None
    label_encoder = None

app = Flask(__name__)
CORS(app)

# ── Hidden Keywords ────────────────────────────────────────────────────────────
# If ANY of these phrases appear anywhere in the spoken text → EMERGENCY
HIDDEN_KEYWORDS = [
    "did you feed the cat",
    "feed the cat",
    "time to dinner",
    "what time is dinner",
    "did you water the plants",
    "water the plants",
    "have you seen my keys",
    "seen my keys",
    "is the door locked",
    "door locked",
]

def check_hidden_keywords(text: str) -> str | None:
    """
    Returns the matched keyword if found in text, else None.
    Case-insensitive, partial match allowed.
    """
    text_lower = text.lower().strip()
    for kw in HIDDEN_KEYWORDS:
        if kw in text_lower:
            return kw
    return None


# ── OpenCV Wave Detector ───────────────────────────────────────────────────────
class OpenCVWaveDetector:
    def __init__(self):
        self.positions       = []
        self.HISTORY_SECONDS = 2.5
        self.MIN_CROSSINGS   = 2
        self.CENTRE_DEADBAND = 0.05
        self.last_x          = 0.5
        self.last_y          = 0.5
        self.last_time       = time.time()

    def process_frame_native(self, frame):
        h, w, _ = frame.shape

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        _, thresh = cv2.threshold(blurred, 80, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        detected_norm_x = None
        detected_norm_y = None
        hand_box        = None

        if contours:
            large = max(contours, key=cv2.contourArea)
            if cv2.contourArea(large) > 5000:
                M = cv2.moments(large)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detected_norm_x = cx / w
                    detected_norm_y = cy / h
                    hand_box        = cv2.boundingRect(large)

        gesture = "IDLE"
        now     = time.time()

        if detected_norm_x is not None:
            self.positions.append((now, detected_norm_x))

        cutoff        = now - self.HISTORY_SECONDS
        self.positions = [(t, x) for t, x in self.positions if t >= cutoff]

        if len(self.positions) >= 4:
            xs     = [x for _, x in self.positions]
            centre = 0.5
            above  = [x > centre + self.CENTRE_DEADBAND for x in xs]
            below  = [x < centre - self.CENTRE_DEADBAND for x in xs]

            crossings = sum(
                1 for i in range(1, len(xs))
                if (above[i] != above[i-1]) or (below[i] != below[i-1])
            )

            if crossings >= self.MIN_CROSSINGS:
                recent = [x for x in xs[-6:]
                          if abs(x - centre) > self.CENTRE_DEADBAND]
                if len(recent) >= 2:
                    gesture = "WAVE_RIGHT" if recent[-1] > recent[0] else "WAVE_LEFT"
                else:
                    gesture = "WAVE_LEFT"

        dt     = max(now - self.last_time, 0.001)
        curr_x = detected_norm_x if detected_norm_x is not None else 0.5
        curr_y = detected_norm_y if detected_norm_y is not None else 0.5
        vx     = (curr_x - self.last_x) / dt
        vy     = (curr_y - self.last_y) / dt

        mock_features = {
            "ax": float(curr_x), "ay": float(curr_y), "az": 0.0,
            "gx": float(vx),     "gy": float(vy),     "gz": 0.0
        }

        self.last_x    = curr_x
        self.last_y    = curr_y
        self.last_time = now

        return gesture, hand_box, mock_features


wave_detector   = OpenCVWaveDetector()
_cooldown_until = 0.0
_state_lock     = threading.Lock()
recognizer      = sr.Recognizer()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Rescue AI Gesture + Speech Detection API Running"})


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

        detected, hand_box, mock_features = wave_detector.process_frame_native(frame)

        gesture_label      = "IDLE"
        is_emergency       = False

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

        if hand_box is not None:
            bx, by, bw, bh = hand_box
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
            cv2.circle(frame, (bx+bw//2, by+bh//2), 5, (255, 0, 0), -1)

        predicted_gesture  = "IDLE"
        encoded_prediction = 0

        if model is not None and label_encoder is not None:
            try:
                input_df           = pd.DataFrame([mock_features],
                                                  columns=["ax","ay","az","gx","gy","gz"])
                prediction         = model.predict(input_df)
                encoded_prediction = int(round(prediction[0]))
                encoded_prediction = max(0, min(encoded_prediction,
                                                len(label_encoder.classes_) - 1))
                predicted_gesture  = str(label_encoder.inverse_transform(
                                         [encoded_prediction])[0])
            except Exception:
                pass

        colour = (0, 0, 255) if is_emergency else (0, 255, 0)
        label  = "EMERGENCY!" if is_emergency else gesture_label
        cv2.putText(frame, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        out_b64   = base64.b64encode(buffer).decode("utf-8")

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


@app.route("/process-audio", methods=["POST"])
def process_audio():
    """
    Receives a base64-encoded WAV audio clip from the browser mic.
    Transcribes using Google Speech Recognition (free, no API key needed).
    Checks transcript against hidden keywords.
    Returns emergency flag if keyword found.
    """
    try:
        data      = request.get_json(force=True)
        b64_audio = data.get("audio", "")

        if not b64_audio:
            return jsonify({"success": False, "error": "No audio provided"}), 400

        if "," in b64_audio:
            b64_audio = b64_audio.split(",", 1)[1]

        audio_bytes = base64.b64decode(b64_audio)

        # Write to a temp WAV file for SpeechRecognition
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            with sr.AudioFile(tmp_path) as source:
                audio_data = recognizer.record(source)

            transcript = recognizer.recognize_google(audio_data).lower()
        except sr.UnknownValueError:
            transcript = ""
        except sr.RequestError as e:
            return jsonify({
                "success":    True,
                "transcript": "",
                "emergency":  False,
                "keyword":    None,
                "error":      f"Speech service error: {e}"
            })
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        matched_keyword = check_hidden_keywords(transcript)
        is_emergency    = matched_keyword is not None

        return jsonify({
            "success":    True,
            "transcript": transcript,
            "emergency":  is_emergency,
            "keyword":    matched_keyword,
            "source":     "speech"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/keywords", methods=["GET"])
def get_keywords():
    """Returns the list of hidden keywords (for frontend display if needed)."""
    return jsonify({"keywords": HIDDEN_KEYWORDS})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or label_encoder is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500
    try:
        data     = request.get_json()
        required = ["ax", "ay", "az", "gx", "gy", "gz"]
        missing  = [f for f in required if f not in data]
        if missing:
            return jsonify({"success": False, "error": f"Missing: {missing}"}), 400

        input_df           = pd.DataFrame([[float(data[f]) for f in required]],
                                          columns=required)
        prediction         = model.predict(input_df)
        encoded_prediction = int(round(prediction[0]))
        encoded_prediction = max(0, min(encoded_prediction,
                                        len(label_encoder.classes_) - 1))
        predicted_gesture  = str(label_encoder.inverse_transform(
                                  [encoded_prediction])[0])

        return jsonify({
            "success":            True,
            "predicted_gesture":  predicted_gesture,
            "encoded_prediction": encoded_prediction
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
