from flask import Blueprint, Response, jsonify, render_template
from flask_login import login_required, current_user
from collections import Counter
import cv2, torch, threading, time, contextlib, json
from ultralytics import YOLO
from .models import Report
from . import db

# ── extra imports for audio + LLM ──────────────────────────────────
from .audio_whisper import AudioTranscriber
from .report_agent_1_5 import InterviewAnalyzer

# ── device / models ────────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"
torch.backends.cudnn.benchmark = True

model_face    = YOLO("website/yolo_model/YOLO11_10B_face.pt")
model_emotion = YOLO("website/yolo_model/YOLO11_20B_emotion.pt")
for m in (model_face, model_emotion):
    m.model.to(DEVICE).eval()
    if USE_FP16:
        m.model.half()

# ── audio + LLM singletons ────────────────────────────────────────
audio_transcriber = AudioTranscriber()
analyzer = InterviewAnalyzer(device=DEVICE)

# ── global state ──────────────────────────────────────────────────
emotion_log: list[str] = []
streaming:   bool      = False

video = Blueprint("video", __name__)

# ─────────────────────────  ROUTES  ───────────────────────────────
@video.route("/video")
@login_required
def video_page():
    return render_template("video.html")


@video.route("/video_feed")
@login_required
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@video.route("/start_stream")
@login_required
def start_stream():
    global streaming
    streaming = True
    emotion_log.clear()
    audio_transcriber.start()
    return jsonify({"status": "started"})


@video.route("/stop_stream")
@login_required
def stop_stream():
    """
    1. Corta el bucle de vídeo inmediatamente.
    2. Para el hilo de audio (bloquea ≤2 s).
    3. Lanza la generación del informe LLM en un hilo de fondo.
    4. Devuelve la respuesta al navegador sin esperar a Phi-1.5.
    """
    global streaming
    streaming = False                    
    
    try:                                  
        audio_transcriber.stop()
    except Exception as e:
        print(f"[ERROR] Audio stop failed: {e}")

    threading.Thread(target=_run_llm_report, daemon=True).start()

    return jsonify({"status": "stopped"})  


@video.route("/get_report")
@login_required
def get_report():
    global emotion_log
    summary = dict(Counter(emotion_log))
    emotion_log.clear()

    db.session.add(Report(
        data=json.dumps(summary),
        user_id=current_user.id
    ))
    db.session.commit()
    return jsonify(summary)


def _run_llm_report() -> None:
    """
    Genera data/report_phi.txt sin bloquear la ruta /stop_stream.
    """
    try:
        report = analyzer.analyze_interview("data/transcripts.csv")
        with open("data/report_phi.txt", "w", encoding="utf-8") as fh:
            fh.write(report)
        print("[INFO] LLM report saved to data/report_phi.txt")
    except Exception as e:
        print(f"[WARNING] LLM report failed: {e}")


class Camera:
    """Threaded capture that always returns the freshest frame."""
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock, self.frame, self.live = threading.Lock(), None, True
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.live and self.cap.isOpened():
            ok, frm = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frm

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.live = False
        time.sleep(0.1)
        self.cap.release()


def generate_frames():
    """Video loop; exits quickly when streaming flag flips to False."""
    global streaming, emotion_log
    cam = Camera()
    stream = torch.cuda.Stream() if DEVICE == "cuda" else None

    try:
        with torch.no_grad():
            while streaming:
                frm = cam.read()
                if frm is None:
                    time.sleep(0.005)
                    continue

                frm = cv2.resize(frm, (640, 480))

                with (torch.cuda.stream(stream)
                      if stream else contextlib.nullcontext()):
                    faces = model_face(frm, device=DEVICE,
                                       half=USE_FP16, verbose=False)[0]

                crops, boxes = [], []
                for b in faces.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    crop = frm[y1:y2, x1:x2]
                    if crop.size:
                        crops.append(crop)
                        boxes.append((x1, y1, x2, y2))

                if crops:
                    emos = model_emotion(crops, device=DEVICE,
                                         half=USE_FP16, verbose=False)
                    for (x1, y1, x2, y2), res in zip(boxes, emos):
                        best = res.boxes[0]
                        label = model_emotion.names[int(best.cls[0])]
                        conf  = float(best.conf[0])

                        emotion_log.append(label)
                        cv2.rectangle(frm, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frm, f"{label} {conf:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 255, 0), 2)

                ok, buf = cv2.imencode(".jpg", frm,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                           buf.tobytes() + b"\r\n")
    finally:
        cam.release()
