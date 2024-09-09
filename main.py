import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model

# Load model yang sudah dilatih
model = load_model('model.h5')

# Inisialisasi MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Fungsi untuk menyimpan gambar
def save_image(image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    cv2.imwrite(filename, image)
    print(f"Image saved as {filename}")

# Fungsi untuk mengubah landmark tangan menjadi array numpy untuk input ke model
def landmarks_to_array(hand_landmarks):
    return np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks], dtype=np.float32).flatten()

# Mulai video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Konversi gambar ke RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Proses deteksi tangan
    results = hands.process(image)

    # Kembalikan gambar ke BGR untuk menampilkan
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Ubah landmark menjadi array untuk input model
            input_data = landmarks_to_array(hand_landmarks.landmark)
            input_data = np.expand_dims(input_data, axis=0)

            # Prediksi dengan model
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)

            # Klasifikasi gerakan
            if predicted_class == 0:
                signal = "Open Hand"
            elif predicted_class == 1:
                signal = "Thumb Folded"
            elif predicted_class == 2:
                signal = "SOS DETECTION"
                save_image(image)  # Simpan gambar jika terdeteksi SOS
            else:
                signal = "Unknown"

            # Tampilkan klasifikasi gerakan di layar
            cv2.putText(image, signal, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Tampilkan hasil
    cv2.imshow('Hand Signal Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
