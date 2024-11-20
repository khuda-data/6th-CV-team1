import cv2
import numpy as np
import time
import mediapipe as mp

# YOLOv4-tiny 설정
config_path = r"C:\Users\kangmin\Desktop\yolov4-tiny.cfg"
weights_path = r"C:\Users\kangmin\Desktop\yolov4-tiny.weights"
coco_names_path = r"C:\Users\kangmin\Desktop\coco.names"

with open(coco_names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# 랜드마크 인덱스 (코)
nose_indices = [1, 2, 98, 168, 195, 5]

# 웹캠 초기화
cap = cv2.VideoCapture(0)

# 오버레이할 이미지 경로
overlay_image_path = r"C:\Users\kangmin\Desktop\aa.png"
overlay_image = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)  # 알파 채널 포함

# 오버레이 이미지 크기 가져오기
overlay_h, overlay_w, _ = overlay_image.shape

# 오버레이 이미지 반전 여부 변수
overlay_flipped = False
last_flip_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 좌우 반전
    frame = cv2.flip(frame, 1)

    # YOLOv4-tiny 모델을 통한 사람 감지
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    height, width, channels = frame.shape
    boxes, confidences, class_ids = [], [], []
    left_area = 0
    right_area = 0

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 0: 사람 클래스
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x1 = int(center_x - w / 2)
                y1 = int(center_y - h / 2)

                boxes.append([x1, y1, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # 바운딩 박스 중심 x 좌표
                if center_x < width // 2:
                    left_area += w * h
                else:
                    right_area += w * h

                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(frame, f'{classes[class_id]} {round(confidence, 2)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 일정 시간 동안 오버레이 이미지 반전 여부를 유지
    if time.time() - last_flip_time > 1:  # 1초마다 체크
        if right_area > left_area and not overlay_flipped:
            overlay_image = cv2.flip(overlay_image, 1)  # 오른쪽에 사람이 많으면 반전
            overlay_flipped = True
            last_flip_time = time.time()  # 반전 시간을 기록
        elif left_area > right_area and overlay_flipped:
            overlay_image = cv2.flip(overlay_image, 1)  # 왼쪽에 사람이 많으면 반전 해제
            overlay_flipped = False
            last_flip_time = time.time()  # 반전 시간을 기록

    # Mediapipe를 통한 얼굴 랜드마크 감지
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # 웹캠 크기 가져오기
    frame_h, frame_w, _ = frame.shape

    # 오버레이 이미지 기본 크기 계산
    scale = min(frame_w / overlay_w, frame_h / overlay_h)  # 오버레이 이미지를 프레임 크기에 맞춤
    new_w = int(overlay_w * scale)
    new_h = int(overlay_h * scale)

    # Depth 기반 크기 조정 비율 초기화
    resize_ratio = 1.0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 코의 Depth 값 저장
            nose_depths = []

            for id, lm in enumerate(face_landmarks.landmark):
                x, y = int(lm.x * frame_w), int(lm.y * frame_h)
                z = lm.z  # Depth 값

                # 코 랜드마크 확인 및 저장
                if id in nose_indices:
                    nose_depths.append(z)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # 코 평균 Depth 계산 (변환된 값)
            nose_avg_depth = sum(nose_depths) / len(nose_depths) if nose_depths else 0
            nose_avg_depth_adjusted = (100 * nose_avg_depth + 10) if nose_depths else 0

            # Depth 값을 화면에 출력
            cv2.putText(frame, f"Nose Avg Depth: {nose_avg_depth_adjusted:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Depth 값 기반으로 리사이즈 비율 결정
            resize_ratio = 1.0  # 기본값
            if 6.5 <= nose_avg_depth_adjusted < 7.0:
                resize_ratio = 0.95
            elif 7.0 <= nose_avg_depth_adjusted < 7.5:
                resize_ratio = 0.90
            elif 7.5 <= nose_avg_depth_adjusted < 8.0:
                resize_ratio = 0.85
            elif 8.0 <= nose_avg_depth_adjusted < 8.5:
                resize_ratio = 0.80
            elif 8.5 <= nose_avg_depth_adjusted < 9.0:
                resize_ratio = 0.75
            elif 9.0 <= nose_avg_depth_adjusted < 9.5:
                resize_ratio = 0.70
            elif 9.5 <= nose_avg_depth_adjusted < 10.0:
                resize_ratio = 0.65
            elif 10.0 <= nose_avg_depth_adjusted < 10.5:
                resize_ratio = 0.60
            elif 10.5 <= nose_avg_depth_adjusted < 11.0:
                resize_ratio = 0.55
            elif 11.0 <= nose_avg_depth_adjusted <= 11.5:
                resize_ratio = 0.50

    # 오버레이 이미지 리사이즈
    adjusted_w = int(new_w * resize_ratio)
    adjusted_h = int(new_h * resize_ratio)
    resized_overlay = cv2.resize(overlay_image, (adjusted_w, adjusted_h), interpolation=cv2.INTER_AREA)

    # 이미지의 시작 좌표 계산 (화면 중앙에 표시)
    start_x = (frame_w - adjusted_w) // 2
    start_y = (frame_h - adjusted_h) // 2

    # ROI 선택 및 이미지 합성
    for c in range(0, 3):  # BGR 채널 합성
        frame[start_y:start_y + adjusted_h, start_x:start_x + adjusted_w, c] = \
            resized_overlay[:, :, c] * (resized_overlay[:, :, 3] / 255.0) + \
            frame[start_y:start_y + adjusted_h, start_x:start_x + adjusted_w, c] * (1.0 - resized_overlay[:, :, 3] / 255.0)

    cv2.imshow('Dynamic Overlay Based on Person and Face', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()
