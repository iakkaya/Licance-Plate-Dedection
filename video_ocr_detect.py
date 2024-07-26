from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR

# OCR kütüphanesini başlat
ocr = PaddleOCR(use_angle_cls=True, lang='tr')

# Modelleri yükle
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO(r'licence_palte_son.pt')

# Video kaynağını başlat
video_path = r'automatic-number-plate-recognition-python-yolov8-main\video.mp4'  # Video dosyası yolunu buraya girin
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Arabayı tespit et
    detections = coco_model(frame)[0]

    # Tespit edilen arabada plakayı tespit et
    license_plates = license_plate_detector(frame)[0]

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        plate_frame = frame[int(y1):int(y2), int(x1):int(x2)]
        
        # Plaka görüntüsünü gri tonlamaya çevir
        gray_plate_frame = cv2.cvtColor(plate_frame, cv2.COLOR_BGR2GRAY)

        # Bulanıklık gidermek için bulanıklık uygulayın (opsiyonel, duruma göre ayarlayın)
        gray_plate_frame = cv2.GaussianBlur(gray_plate_frame, (3, 3), 0)

        # OCR ile plakayı oku
        ocr_results = ocr.ocr(gray_plate_frame)
        plate = None
        for line in ocr_results:
            try:
                plate = line[0][1][0]
                break
            except:
                pass

        # Tespit edilen koordinatları kullanarak dikdörtgen çiz
        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))
        color = (255, 0, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, color, thickness)

        # Okunan plakayı yazdır
        if plate:
            print(f"Tespit edilen plaka: {plate}")

    # Sonucu göster
    cv2.imshow('Tespit edilen plaka', gray_plate_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()