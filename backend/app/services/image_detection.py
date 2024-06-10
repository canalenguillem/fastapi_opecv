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

def detect_faces(image_path, scaleFactor=1.1, minNeighbors=5, margin=30):
    # Cargar la imagen
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Cargar el clasificador de Haar
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detectar los rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    # Convertir imagen a blanco y negro
    bw_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    detected_faces = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]  # Diferentes colores para los contornos

    for i, (x, y, w, h) in enumerate(faces):
        color = colors[i % len(colors)]
        color_hex = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])  # Convertir a formato hex
        
        # Ampliar los límites del recorte
        x_margin = max(0, x - margin)
        y_margin = max(0, y - margin)
        w_margin = min(img.shape[1], x + w + margin) - x_margin
        h_margin = min(img.shape[0], y + h + margin) - y_margin
        
        # Dibujar el contorno en la imagen blanco y negro
        cv2.rectangle(bw_img, (x, y), (x+w, y+h), color, 2)
        
        # Recortar el rostro y agregar contorno de color
        face = img[y_margin:y_margin+h_margin, x_margin:x_margin+w_margin]
        face_image = PILImage.fromarray(face)
        draw = ImageDraw.Draw(face_image)
        draw.rectangle([0, 0, w_margin, h_margin], outline=color, width=2)

        detected_faces.append((face_image, x_margin, y_margin, w_margin, h_margin, color_hex))

    # Guardar la imagen en blanco y negro con contornos de colores
    bw_image_path = os.path.join("uploads", "bw_" + os.path.basename(image_path))
    bw_image = PILImage.fromarray(bw_img)
    bw_image.save(bw_image_path)

    return bw_image_path, detected_faces
