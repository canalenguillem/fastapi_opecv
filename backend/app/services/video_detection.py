import cv2
from PIL import Image as PILImage, ImageDraw
from fastapi import UploadFile
import os
import numpy as np

def save_uploaded_file(file: UploadFile):
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    return file_location

def detect_faces_in_video(video_path, video_id, scaleFactor=1.1, minNeighbors=5, margin=30):
    # Cargar el clasificador de Haar
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detected_faces = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]  # Diferentes colores para los contornos

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        for i, (x, y, w, h) in enumerate(faces):
            color = colors[i % len(colors)]
            color_hex = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])  # Convertir a formato hex

            # Ampliar los límites del recorte
            x_margin = max(0, x - margin)
            y_margin = max(0, y - margin)
            w_margin = min(frame.shape[1], x + w + margin) - x_margin
            h_margin = min(frame.shape[0], y + h + margin) - y_margin

            # Recortar el rostro
            face = frame[y_margin:y_margin+h_margin, x_margin:x_margin+w_margin]
            face_image = PILImage.fromarray(face)
            draw = ImageDraw.Draw(face_image)
            draw.rectangle([0, 0, w_margin, h_margin], outline=color, width=2)

            face_path = f"faces/{video_id}_{frame_count}_{i}.jpg"
            face_image.save(face_path)

            detected_faces.append({
                "x": x_margin,
                "y": y_margin,
                "w": w_margin,
                "h": h_margin,
                "frame": frame_count,
                "color": color_hex,
                "path": face_path
            })

        frame_count += 1

    cap.release()
    return detected_faces
