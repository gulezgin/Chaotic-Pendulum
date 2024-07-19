import numpy as np
import cv2

# Kamera yakalama
ip_camera_url = 'http://192.168.16.195:8080/video'  # IP kamera URL'sini buraya girin
cap = cv2.VideoCapture(ip_camera_url)

def detect_circle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray.shape[0] // 8,
                               param1=100, param2=30, minRadius=1, maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # İlk daire için farklı bir renk kullanın
            if i is circles[0, 0]:
                cv2.circle(frame, (i[0], i[1]), i[2], (255, 0, 0), 2)  # Mavi
                cv2.circle(frame, (i[0], i[1]), 2, (255, 0, 0), 3)  # Mavi
            else:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Yeşil
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)  # Kırmızı
        return circles[0][0]  # İlk dairenin koordinatları
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamıyor.")
        break

    coords = detect_circle(frame)
    if coords is not None:
        print(f"Circle detected at: {coords[0]}, {coords[1]}")
    else:
        print("Circle not detected.")

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
