from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR

# OCR kütüphanesini başlat
ocr = PaddleOCR(use_angle_cls=True, lang='tr')

# Modelleri yükle
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO(r'licence_palte_son.pt')

# Görüntüyü oku
img = cv2.imread(r'automatic-number-plate-recognition-python-yolov8-main/images/i-1.jpg')

# Arabayı tespit et
detections = coco_model(img)[0]

# Tespit edilen arabada plakayı tespit et
license_plates = license_plate_detector(img)[0]

for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = license_plate
    plate_frame = img[int(y1):int(y2), int(x1):int(x2)]
    
    # OCR ile plakayı oku
    ocr_results = ocr.ocr(plate_frame)
    plate = None
    for line in ocr_results:
        try:
            s1 = line[0]
            s2 = s1[1]
            plate_code = s2[0]
            #tr_code = line[3][1][0]
            #first_code = line[0][1][0]
            #second_code = line[1][1][0]
            #third_code = line[2][1][0]
            #plate = tr_code + " " + first_code + " " + second_code + " " +third_code
            plate = plate_code
            break
        except:
            pass

    # Tespit edilen koordinatları kullanarak dikdörtgen çiz
    start_point = (int(x1), int(y1))
    end_point = (int(x2), int(y2))
    color = (255, 0, 0)
    thickness = 2
    #img = cv2.rectangle(img, start_point, end_point, color, thickness)
    img = plate_frame
    # Okunan plakayı yazdır
    if plate:
        print(f"Tespit edilen plaka: {plate}")

# Sonucu göster
cv2.imshow("Tespit edilen plaka", img)
cv2.waitKey(0)
cv2.destroyAllWindows()