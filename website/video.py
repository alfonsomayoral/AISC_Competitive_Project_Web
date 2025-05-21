# website/video.py  (versión optimizada GPU)
from flask import Blueprint, Response, jsonify, render_template
import cv2
from ultralytics import YOLO
from collections import Counter
from flask_login import login_required, current_user
import torch                                          
import json
from . import db
from .models import Report
from .audio_whisper import AudioTranscriber

video = Blueprint('video', __name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_HALF = DEVICE == 'cuda'                           
print(f"[INFO] Inference device: {DEVICE}")


model_face    = YOLO('website/yolo_model/YOLO11_10B_face.pt')
model_emotion = YOLO('website/yolo_model/YOLO11_20B_emotion.pt')

for m in (model_face, model_emotion):
    m.model.to(DEVICE).eval()
    if USE_HALF:
        m.model.half()

emotion_log = []
streaming   = False
audio_transcriber = AudioTranscriber()


@video.route('/video')
@login_required
def video_page():
    return render_template("video.html")

@video.route('/video_feed')
@login_required
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@video.route('/start_stream')
@login_required
def start_stream():
    global streaming, audio_transcriber
    try:
        audio_transcriber.start()
        streaming = True
        return jsonify({'status': 'started'})
    except Exception as e:
        print(f"[ERROR] Failed to start audio: {e}")
        streaming = False
        return jsonify({'status': 'error', 'message': str(e)}), 500

@video.route('/stop_stream')
@login_required
def stop_stream():
    global streaming, audio_transcriber
    try:
        audio_transcriber.stop()
        return jsonify({'status': 'stopped'})
    except Exception as e:
        print(f"[ERROR] Error stopping audio: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        streaming = False

@video.route('/get_report')
@login_required
def get_report():
    global emotion_log
    summary = dict(Counter(emotion_log))
    emotion_log.clear()

    new_report = Report(
        data=json.dumps(summary),
        user_id=current_user.id
    )
    db.session.add(new_report)
    db.session.commit()

    print(f"[INFO] Report saved for user {current_user.id}")
    return jsonify(summary)

def generate_frames():
    global streaming, emotion_log
    print("[INFO] Starting video capture...")
    # CAP_DSHOW + buffer=1 → menos latencia en Windows               
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Could not access the webcam.")
        return

    with torch.no_grad():                                           
        while streaming and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame capture failed.")
                break

            frame = cv2.resize(frame, (640, 480))

            face_results = model_face(
                frame, device=DEVICE, half=USE_HALF, verbose=False   
            )[0]

            for box in face_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                emotion_results = model_emotion(
                    face_crop, device=DEVICE, half=USE_HALF, verbose=False
                )[0]

                if emotion_results.boxes is not None and len(emotion_results.boxes) > 0:
                    best = emotion_results.boxes[0]
                    label = model_emotion.names[int(best.cls[0])]
                    conf  = float(best.conf[0])

                    emotion_log.append(label)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                    )

            ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

    print("[INFO] Releasing video capture.")
    cap.release()
