import cv2
import mediapipe as mp
from datetime import datetime

# Inisialisasi MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Status untuk menyimpan tahapan gerakan
gesture_stage = 0

# Fungsi untuk mendeteksi gerakan tangan
def detect_hand_signal(hand_landmarks):
    global gesture_stage

    # Landmarks for fingers A-E (thumb to pinky)
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]
    ring_tip = hand_landmarks[16]
    pinky_tip = hand_landmarks[20]

    # Get knuckle landmarks
    thumb_knuckle = hand_landmarks[3]
    index_knuckle = hand_landmarks[6]
    middle_knuckle = hand_landmarks[10]
    ring_knuckle = hand_landmarks[14]
    pinky_knuckle = hand_landmarks[18]

    # Stage 1: All fingers (A to E) open
    all_fingers_open = (
        thumb_tip.x < thumb_knuckle.x and
        index_tip.y < index_knuckle.y and
        middle_tip.y < middle_knuckle.y and
        ring_tip.y < ring_knuckle.y and
        pinky_tip.y < pinky_knuckle.y
    )

    # Stage 2: Only finger E (pinky) folds
    pinky_folded = pinky_tip.y > pinky_knuckle.y

    # Stage 3: Fingers A to D (thumb to ring) close over E (pinky)
    fingers_a_d_folded_over_e = (
        thumb_tip.x > thumb_knuckle.x and
        index_tip.y > index_knuckle.y and
        middle_tip.y > middle_knuckle.y and
        ring_tip.y > ring_knuckle.y and
        pinky_folded
    )

    # Deteksi tahapan gerakan
    if gesture_stage == 0 and all_fingers_open:
        gesture_stage = 1
    elif gesture_stage == 1 and pinky_folded and not all_fingers_open:
        gesture_stage = 2
    elif gesture_stage == 2 and fingers_a_d_folded_over_e:
        gesture_stage = 3
        return "SOS DETECTION"

    return "No SOS"

# Fungsi untuk menyimpan gambar
def save_image(image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    cv2.imwrite(filename, image)
    print(f"Image saved as {filename}")

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
            signal = detect_hand_signal(hand_landmarks.landmark)
            cv2.putText(image, signal, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if signal == "SOS DETECTION":
                save_image(image)
                gesture_stage = 0  # Reset tahapan setelah capture

    # Tampilkan hasil
    cv2.imshow('Hand Signal Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
