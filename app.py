import joblib
import pandas as pd
import numpy as np
import cv2
import base64
import time
import threading
import warnings
import os

from flask import Flask, request, jsonify
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ── ML Models ─────────────────────────────────────────────────────────────────
try:
    model         = joblib.load("knn_regressor_model.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
except Exception as e:
    print(f"CRITICAL Warning during model loading: {e}")
    model = None
    label_encoder = None

app = Flask(__name__)
CORS(app)

# ── OpenCV Wave & Motion Tracker ──────────────────────────────────────────────
class OpenCVWaveDetector:
    def __init__(self):
        self.positions = []       # List of (timestamp, norm_x)
        self.HISTORY_SECONDS = 2.5 # Time window for checking swings
        self.MIN_CROSSINGS   = 2   # Direction changes to count as a wave
        self.CENTRE_DEADBAND = 0.05
        
        # Keep track of previous coordinates to mock sensor/landmark deltas for KNN
        self.last_x = 0.5
        self.last_y = 0.5
        self.last_time = time.time()

    def process_frame_native(self, frame):
        """
        Detects the main moving object (hand) using thresholding/contour tracking,
        tracks its normalized X coordinate, and calculates gesture state.
        """
        h, w, _ = frame.shape
        
        # 1. Convert to grayscale and blur to remove image noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # 2. Dynamic Adaptive Thresholding to segment skin/foreground silhouettes
        _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. Find structural contours in the image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_norm_x = None
        detected_norm_y = None
        hand_box = None
        
        if contours:
            # Assume the largest prominent foreground shape is the hand/arm
            large_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(large_contour) > 5000: # Ignore tiny random particles
                M = cv2.moments(large_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Normalize positions between 0.0 and 1.0
                    detected_norm_x = cx / w
                    detected_norm_y = cy / h
                    hand_box = cv2.boundingRect(large_contour)

        # 4. Wave Algorithm Evaluation using tracked horizontal centroids
        gesture = "IDLE"
        now = time.time()
        
        if detected_norm_x is not None:
            self.positions.append((now, detected_norm_x))
            
        # Clean history window
        cutoff = now - self.HISTORY_SECONDS
        self.positions = [(t, x) for t, x in self.positions if t >= cutoff]
        
        if len(self.positions) >= 4:
            xs = [x for _, x in self.positions]
            centre = 0.5
            above = [x > centre + self.CENTRE_DEADBAND for x in xs]
            below = [x < centre - self.CENTRE_DEADBAND for x in xs]
            
            crossings = sum(
                1 for i in range(1, len(xs))
                if (above[i] != above[i-1]) or (below[i] != below[i-1])
            )
            
            if crossings >= self.MIN_CROSSINGS:
                recent = [x for x in xs[-6:] if abs(x - centre) > self.CENTRE_DEADBAND]
                if len(recent) >= 2:
                    gesture = "WAVE_RIGHT" if recent[-1] > recent[0] else "WAVE_LEFT"
                else:
                    gesture = "WAVE_LEFT"

        # 5. Extract speed deltas to feed your model's required (ax, ay, az, gx, gy, gz) variables safely
        dt = max(now - self.last_time, 0.001)
        curr_x = detected_norm_x if detected_norm_x is not None else 0.5
        curr_y = detected_norm_y if detected_norm_y is not None else 0.5
        
        vx = (curr_x - self.last_x) / dt
        vy = (curr_y - self.last_y) / dt
        
        # Synthesize equivalent spatial feature inputs
        mock_features = {
            "ax": float(curr_x), "ay": float(curr_y), "az": 0.0,
            "gx": float(vx),     "gy": float(vy),     "gz": 0.0
        }
        
        # Save track history state
        self.last_x = curr_x
        self.last_y = curr_y
        self.last_time = now
        
        return gesture, hand_box, mock_features


wave_detector   = OpenCVWaveDetector()
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
    runs pure OpenCV tracking + algorithmic wave detection,
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

        # Mirror frame alignment
        frame = cv2.flip(frame, 1)

        # Process hand spatial metrics purely using OpenCV tracking
        detected, hand_box, mock_features = wave_detector.process_frame_native(frame)

        gesture_label = "IDLE"
        is_emergency  = False

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

        # Visual bounding box over the hand tracked by OpenCV contours
        if hand_box is not None:
            bx, by, bw, bh = hand_box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.circle(frame, (bx + bw//2, by + bh//2), 5, (255, 0, 0), -1)

        # Machine Learning classification prediction running over spatial properties
        predicted_gesture  = "IDLE"
        encoded_prediction = 0

        if model is not None and label_encoder is not None:
            input_df = pd.DataFrame([mock_features], columns=["ax","ay","az","gx","gy","gz"])
            prediction = model.predict(input_df)
            encoded_prediction = int(round(prediction[0]))
            encoded_prediction = max(0, min(encoded_prediction, len(label_encoder.classes_) - 1))
            predicted_gesture  = str(label_encoder.inverse_transform([encoded_prediction])[0])

        # Write text annotations
        colour = (0, 0, 255) if is_emergency else (0, 255, 0)
        label  = "EMERGENCY!" if is_emergency else gesture_label
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)

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
        predicted_gesture  = str(label_encoder.inverse_transform([encoded_prediction])[0])

        return jsonify({
            "success":            True,
            "predicted_gesture":  str(predicted_gesture),
            "encoded_prediction": int(encoded_prediction)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
