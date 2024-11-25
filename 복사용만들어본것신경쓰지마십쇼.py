from flask import Flask, Response, render_template, request, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
import os

app = Flask(__name__)

# YOLO 설정
config_path = r"C:\Users\82102\Desktop\photoism\yolov4-tiny.cfg"
weights_path = r"C:\Users\82102\Desktop\photoism\yolov4-tiny.weights"
coco_names_path = r"C:\Users\82102\Desktop\photoism\coco.names"

with open(coco_names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# 준비된 이미지 경로
prepared_images_dir = r"C:\Users\82102\Desktop\photoism\input"
prepared_images = {
    50: cv2.imread(os.path.join(prepared_images_dir, "cropped_50.png"), cv2.IMREAD_UNCHANGED),
    55: cv2.imread(os.path.join(prepared_images_dir, "cropped_55.png"), cv2.IMREAD_UNCHANGED),
    60: cv2.imread(os.path.join(prepared_images_dir, "cropped_60.png"), cv2.IMREAD_UNCHANGED),
    65: cv2.imread(os.path.join(prepared_images_dir, "cropped_65.png"), cv2.IMREAD_UNCHANGED),
    70: cv2.imread(os.path.join(prepared_images_dir, "cropped_70.png"), cv2.IMREAD_UNCHANGED),
    75: cv2.imread(os.path.join(prepared_images_dir, "cropped_75.png"), cv2.IMREAD_UNCHANGED),
    80: cv2.imread(os.path.join(prepared_images_dir, "cropped_80.png"), cv2.IMREAD_UNCHANGED),
    85: cv2.imread(os.path.join(prepared_images_dir, "cropped_85.png"), cv2.IMREAD_UNCHANGED),
    90: cv2.imread(os.path.join(prepared_images_dir, "cropped_90.png"), cv2.IMREAD_UNCHANGED),
    95: cv2.imread(os.path.join(prepared_images_dir, "cropped_95.png"), cv2.IMREAD_UNCHANGED),
    100: cv2.imread(os.path.join(prepared_images_dir, "cropped_100.png"), cv2.IMREAD_UNCHANGED),
}

# 'choose' 폴더의 이미지 파일 목록
choose_images_dir = r"C:\Users\82102\Desktop\photoism\static\choose"
choose_images = [os.path.join(choose_images_dir, f) for f in os.listdir(choose_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Flask 라우트
@app.route('/')
def choose_frame():
    images = [url_for('static', filename=f'choose/{os.path.basename(img)}') for img in choose_images]
    return render_template('choose.html', images=images)

@app.route('/webcam')
def start_webcam():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    global last_frame  # 가장 최근의 프레임
    if last_frame is not None:
        # 파일 저장 경로를 static/captures로 설정
        file_path = os.path.join(r"C:\Users\82102\Desktop\photoism\static\captures", "capture.jpg")
        cv2.imwrite(file_path, last_frame)
        print(f"Photo saved at {file_path}")
        return "Photo captured successfully", 200
    return "No frame available for capture", 400

@app.route('/show_photo')
def show_photo():
    return render_template('show.html')

@app.route('/scan')
def scan_page():
    return render_template('scan.html')

@app.route('/select_frame', methods=['POST'])
def select_frame():
    selected_image = request.form.get('selected_image')
    print(f"Selected Frame: {selected_image}")
    return redirect(url_for('scan_page'))

last_frame = None  # 전역 변수로 선언

# OpenCV 비디오 스트림 생성
def generate_frames():
    global last_frame  # 최근 프레임을 저장
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 좌우 반전
        frame = cv2.flip(frame, 1)

        # Mediapipe 얼굴 랜드마크 탐지
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # 화면 크기 정보
        frame_h, frame_w, _ = frame.shape

        # Depth 기반 오버레이 선택
        selected_overlay = prepared_images[100]
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose_depths = [lm.z for id, lm in enumerate(face_landmarks.landmark) if id in [1, 2, 98, 168, 195, 5]]
                nose_avg_depth = sum(nose_depths) / len(nose_depths) if nose_depths else 0
                nose_avg_depth_adjusted = 100 * nose_avg_depth + 10

                if nose_avg_depth_adjusted < 6.8:
                    selected_overlay = prepared_images[50]
                elif 6.8 <= nose_avg_depth_adjusted < 7.02:
                    selected_overlay = prepared_images[55]
                elif 7.02 <= nose_avg_depth_adjusted < 7.24:
                    selected_overlay = prepared_images[60]
                elif 7.24 <= nose_avg_depth_adjusted < 7.46:
                    selected_overlay = prepared_images[65]
                elif 7.46 <= nose_avg_depth_adjusted < 7.68:
                    selected_overlay = prepared_images[70]
                elif 7.68 <= nose_avg_depth_adjusted < 7.9:
                    selected_overlay = prepared_images[75]
                elif 7.9 <= nose_avg_depth_adjusted < 8.2:
                    selected_overlay = prepared_images[80]
                elif 8.2 <= nose_avg_depth_adjusted < 8.34:
                    selected_overlay = prepared_images[85]
                elif 8.34 <= nose_avg_depth_adjusted < 8.45:
                    selected_overlay = prepared_images[90]
                elif 8.45 <= nose_avg_depth_adjusted < 8.53:
                    selected_overlay = prepared_images[95]
                else:
                    selected_overlay = prepared_images[100]

        # 오버레이 리사이즈
        overlay_h, overlay_w, _ = selected_overlay.shape
        scale = min(frame_w / overlay_w, frame_h / overlay_h)
        new_overlay_w = int(overlay_w * scale)
        new_overlay_h = int(overlay_h * scale)
        resized_overlay = cv2.resize(selected_overlay, (new_overlay_w, new_overlay_h), interpolation=cv2.INTER_AREA)

        # YOLO를 사용한 좌우 반전 결정
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        left_area, right_area = 0, 0
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # 사람
                    center_x = int(detection[0] * frame_w)
                    w = int(detection[2] * frame_w)
                    h = int(detection[3] * frame_h)

                    if center_x < frame_w // 2:
                        left_area += w * h
                    else:
                        right_area += w * h

        if right_area < left_area:
            resized_overlay = cv2.flip(resized_overlay, 1)

        # 이미지 합성
        start_x = (frame_w - new_overlay_w) // 2
        start_y = (frame_h - new_overlay_h) // 2
        for c in range(0, 3):
            alpha = resized_overlay[:, :, 3] / 255.0
            frame[start_y:start_y + new_overlay_h, start_x:start_x + new_overlay_w, c] = \
                resized_overlay[:, :, c] * alpha + frame[start_y:start_y + new_overlay_h, start_x:start_x + new_overlay_w, c] * (1 - alpha)

        last_frame = frame.copy()  # 마지막 프레임 저장

        # 프레임 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)