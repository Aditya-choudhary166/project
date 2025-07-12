import cv2
from deepface import DeepFace
import pygame
import os
import time

pygame.mixer.init()


base_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(base_path, "haarcascade_frontalface_default.xml")
music_path = os.path.join(base_path, "music")

face_cascade = cv2.CascadeClassifier(xml_path)


emotion_music = {
    "happy": "tell-me-the-truth.mp3.mp3",
    "sad": "sad.mp3",
    "angry": "angry.mp3",
    "neutral": "brain-implant-cyberpunk-sci-fi-trailer-action-intro-330416.mp3"
}


cap = cv2.VideoCapture(0)
last_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        
        face_crop = frame[y:y+h, x:x+w]
        try:
            result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            
            if emotion != last_emotion and emotion in emotion_music:
                last_emotion = emotion
                pygame.mixer.music.stop()
                music_file = os.path.join(music_path, emotion_music[emotion])
                if os.path.exists(music_file):
                    pygame.mixer.music.load(music_file)
                    pygame.mixer.music.play()
        except Exception as e:
            print("Emotion detection error:", e)

    cv2.imshow("Emotion Music Player", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
