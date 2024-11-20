import cv2
import numpy as np
import time

config_path = r"C:\Users\kangmin\Desktop\yolov4-tiny.cfg"
weights_path = r"C:\Users\kangmin\Desktop\yolov4-tiny.weights"
coco_names_path = r"C:\Users\kangmin\Desktop\coco.names"

with open(coco_names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)
overlay_image_path = r"C:\Users\kangmin\Desktop\aa.png"
overlay_image = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)  # 알파 채널 포함

# 오버레이 이미지 크기 가져오기
overlay_h, overlay_w, _ = overlay_image.shape

# 오버레이 이미지 반전 여부 변수
overlay_flipped = False
last_flip_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

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

    # 오버레이 이미지 크기 조정
    scale = min(width / overlay_w, height / overlay_h)
    new_w = int(overlay_w * scale)
    new_h = int(overlay_h * scale)
    resized_overlay = cv2.resize(overlay_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 오버레이 이미지 배치 위치 계산 (왼쪽에 배치)
    start_x = (width - new_w) // 4  # 왼쪽 1/4 지점에 배치
    start_y = (height - new_h) // 2  # 화면 중앙에 배치

    # 오버레이 이미지 합성
    for c in range(0, 3):  # BGR 채널 합성
        frame[start_y:start_y + new_h, start_x:start_x + new_w, c] = \
            resized_overlay[:, :, c] * (resized_overlay[:, :, 3] / 255.0) + \
            frame[start_y:start_y + new_h, start_x:start_x + new_w, c] * (1.0 - resized_overlay[:, :, 3] / 255.0)

    cv2.imshow("YOLOv4-tiny Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
