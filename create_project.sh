#!/bin/bash

# Crear nueva estructura de directorios
mkdir -p backend/app/api/endpoints
mkdir -p backend/app/api/schemas
mkdir -p backend/app/crud
mkdir -p backend/app/db
mkdir -p backend/app/models
mkdir -p backend/app/services
mkdir -p backend/uploads
mkdir -p backend/faces

# Crear archivos __init__.py para todos los nuevos directorios
touch backend/app/__init__.py
touch backend/app/api/__init__.py
touch backend/app/api/endpoints/__init__.py
touch backend/app/api/schemas/__init__.py
touch backend/app/crud/__init__.py
touch backend/app/db/__init__.py
touch backend/app/models/__init__.py
touch backend/app/services/__init__.py

# Crear archivo de rutas principal
cat <<EOL > backend/app/api/endpoints/pictures.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.db.db import get_db
from app.crud.pictures import save_image, save_faces, save_bw_image, delete_image_and_faces
from app.api.schemas.pictures import ImageCreate, FaceCreate, BwImageCreate
from app.services.image_detection import save_uploaded_file, detect_faces
from app.models.pictures import Image
import hashlib
import os

router = APIRouter()

@router.post("/upload/")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Guardar el archivo subido
    file_location = save_uploaded_file(file)
    
    # Calcular el hash de la imagen
    file_hash = hashlib.md5(open(file_location, "rb").read()).hexdigest()
    
    # Verificar si la imagen ya existe en la base de datos
    if db.query(Image).filter(Image.hash == file_hash).first():
        raise HTTPException(status_code=400, detail="Image already exists")
    
    # Guardar referencia a la imagen en la base de datos
    image_data = ImageCreate(hash=file_hash, path=file_location)
    image = save_image(db=db, image=image_data)
    
    # Detectar rostros en la imagen
    bw_image_path, faces = detect_faces(file_location)
    
    # Guardar la imagen en blanco y negro con contornos
    bw_image_data = BwImageCreate(original_image_id=image.id, path=bw_image_path)
    save_bw_image(db=db, bw_image=bw_image_data)

    # Guardar los rostros detectados
    for i, (face, x, y, w, h, color) in enumerate(faces):
        face_path = os.path.join("faces", f"{image.id}_{i}.jpg")
        face.save(face_path)
        face_data = FaceCreate(image_id=image.id, path=face_path, x=x, y=y, w=w, h=h, color=color)
        save_faces(db=db, face=face_data)
    
    return JSONResponse(content={"image_id": image.id, "faces": len(faces), "bw_image_path": bw_image_path})

@router.delete("/{image_id}")
async def delete_image(image_id: int, db: Session = Depends(get_db)):
    # Eliminar la imagen y los rostros asociados
    delete_image_and_faces(db, image_id)
    return JSONResponse(content={"message": "Image and faces deleted successfully"})
EOL

cat <<EOL > backend/app/api/endpoints/videos.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.db.db import get_db
from app.crud.videos import save_video, save_video_faces, delete_video_and_faces
from app.api.schemas.videos import VideoCreate, VideoFaceCreate
from app.services.video_detection import save_uploaded_file, detect_faces_in_video
from app.models.videos import Video
import hashlib
import os

router = APIRouter()

@router.post("/upload/")
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Guardar el archivo subido
    file_location = save_uploaded_file(file)
    
    # Calcular el hash del vídeo
    file_hash = hashlib.md5(open(file_location, "rb").read()).hexdigest()
    
    # Verificar si el vídeo ya existe en la base de datos
    if db.query(Video).filter(Video.hash == file_hash).first():
        raise HTTPException(status_code=400, detail="Video already exists")
    
    # Guardar referencia al vídeo en la base de datos
    video_data = VideoCreate(hash=file_hash, path=file_location)
    video = save_video(db=db, video=video_data)
    
    # Detectar rostros en el vídeo
    faces = detect_faces_in_video(file_location, video.id)
    
    # Guardar los rostros detectados
    for face in faces:
        face_data = VideoFaceCreate(video_id=video.id, **face)
        save_video_faces(db=db, face=face_data)
    
    return JSONResponse(content={"video_id": video.id, "faces": len(faces)})

@router.delete("/{video_id}")
async def delete_video(video_id: int, db: Session = Depends(get_db)):
    # Eliminar el vídeo y los rostros asociados
    delete_video_and_faces(db, video_id)
    return JSONResponse(content={"message": "Video and faces deleted successfully"})
EOL

# Crear archivo principal de la aplicación
cat <<EOL > backend/app/api/main.py
from fastapi import FastAPI
from app.api.endpoints.pictures import router as pictures_router
from app.api.endpoints.videos import router as videos_router

app = FastAPI()

app.include_router(pictures_router, prefix="/pictures")
app.include_router(videos_router, prefix="/videos")

# Crear las tablas de la base de datos
from app.db.db import engine, Base
Base.metadata.create_all(bind=engine)
EOL

# Crear archivo de esquemas para imágenes
cat <<EOL > backend/app/api/schemas/pictures.py
from pydantic import BaseModel

class ImageBase(BaseModel):
    hash: str
    path: str

class ImageCreate(ImageBase):
    pass

class Image(ImageBase):
    id: int
    class Config:
        orm_mode = True

class FaceBase(BaseModel):
    image_id: int
    path: str
    x: int
    y: int
    w: int
    h: int
    color: str

class FaceCreate(FaceBase):
    pass

class Face(FaceBase):
    id: int
    class Config:
        orm_mode = True

class BwImageBase(BaseModel):
    original_image_id: int
    path: str

class BwImageCreate(BwImageBase):
    pass

class BwImage(BwImageBase):
    id: int
    class Config:
        orm_mode = True
EOL

# Crear archivo de esquemas para vídeos
cat <<EOL > backend/app/api/schemas/videos.py
from pydantic import BaseModel

class VideoBase(BaseModel):
    hash: str
    path: str

class VideoCreate(VideoBase):
    pass

class Video(VideoBase):
    id: int
    class Config:
        orm_mode = True

class VideoFaceBase(BaseModel):
    video_id: int
    x: int
    y: int
    w: int
    h: int
    frame: int
    color: str

class VideoFaceCreate(VideoFaceBase):
    pass

class VideoFace(VideoFaceBase):
    id: int
    class Config:
        orm_mode = True
EOL

# Crear archivo de modelos para imágenes
cat <<EOL > backend/app/models/pictures.py
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from app.db.db import Base

class Image(Base):
    __tablename__ = "images"
    
    id = Column(Integer, primary_key=True, index=True)
    hash = Column(String, unique=True, index=True)
    path = Column(String)
    
    faces = relationship("Face", back_populates="image", cascade="all, delete-orphan")
    bw_image = relationship("BwImage", back_populates="original_image", uselist=False, cascade="all, delete-orphan")

class Face(Base):
    __tablename__ = "faces"
    
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"))
    path = Column(String)
    x = Column(Integer)
    y = Column(Integer)
    w = Column(Integer)
    h = Column(Integer)
    color = Column(String)
    
    image = relationship("Image", back_populates="faces")

class BwImage(Base):
    __tablename__ = "bw_images"

    id = Column(Integer, primary_key=True, index=True)
    original_image_id = Column(Integer, ForeignKey("images.id"))
    path = Column(String)

    original_image = relationship("Image", back_populates="bw_image")
EOL

# Crear archivo de modelos para vídeos
cat <<EOL > backend/app/models/videos.py
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from app.db.db import Base

class Video(Base):
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True, index=True)
    hash = Column(String, unique=True, index=True)
    path = Column(String)
    
    faces = relationship("VideoFace", back_populates="video", cascade="all, delete-orphan")

class VideoFace(Base):
    __tablename__ = "video_faces"
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"))
    x = Column(Integer)
    y = Column(Integer)
    w = Column(Integer)
    h = Column(Integer)
    frame = Column(Integer)
    color = Column(String)
    
    video = relationship("Video", back_populates="faces")
EOL

# Crear archivo de funciones CRUD para imágenes
cat <<EOL > backend/app/crud/pictures.py
from sqlalchemy.orm import Session
from app.models.pictures import Image, Face, BwImage
from app.api.schemas.pictures import ImageCreate, FaceCreate, BwImageCreate
import os

def save_image(db: Session, image: ImageCreate):
    db_image = Image(hash=image.hash, path=image.path)
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return db_image

def save_faces(db: Session, face: FaceCreate):
    db_face = Face(
        image_id=face.image_id, 
        path=face.path, 
        x=face.x, 
        y=face.y, 
        w=face.w, 
        h=face.h, 
        color=face.color
    )
    db.add(db_face)
    db.commit()
    db.refresh(db_face)
    return db_face

def save_bw_image(db: Session, bw_image: BwImageCreate):
    db_bw_image = BwImage(
        original_image_id=bw_image.original_image_id,
        path=bw_image.path
    )
    db.add(db_bw_image)
    db.commit()
    db.refresh(db_bw_image)
    return db_bw_image

def delete_image_and_faces(db: Session, image_id: int):
    image = db.query(Image).filter(Image.id == image_id).first()
    if image:
        # Eliminar archivos de rostros y la imagen
        for face in image.faces:
            if os.path.exists(face.path):
                os.remove(face.path)
        if os.path.exists(image.path):
            os.remove(image.path)
        if image.bw_image and os.path.exists(image.bw_image.path):
            os.remove(image.bw_image.path)
        # Eliminar de la base de datos
        db.delete(image)
        db.commit()
EOL

# Crear archivo de funciones CRUD para vídeos
cat <<EOL > backend/app/crud/videos.py
from sqlalchemy.orm import Session
from app.models.videos import Video, VideoFace
from app.api.schemas.videos import VideoCreate, VideoFaceCreate
import os

def save_video(db: Session, video: VideoCreate):
    db_video = Video(hash=video.hash, path=video.path)
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    return db_video

def save_video_faces(db: Session, face: VideoFaceCreate):
    db_face = VideoFace(
        video_id=face.video_id,
        x=face.x,
        y=face.y,
        w=face.w,
        h=face.h,
        frame=face.frame,
        color=face.color
    )
    db.add(db_face)
    db.commit()
    db.refresh(db_face)
    return db_face

def delete_video_and_faces(db: Session, video_id: int):
    video = db.query(Video).filter(Video.id == video_id).first()
    if video:
        # Eliminar archivos de rostros y el vídeo
        for face in video.faces:
            face_path = f"{face.video_id}_{face.frame}_{face.x}_{face.y}.jpg"
            if os.path.exists(face_path):
                os.remove(face_path)
        if os.path.exists(video.path):
            os.remove(video.path)
        # Eliminar de la base de datos
        db.delete(video)
        db.commit()
EOL

# Crear archivo de detección de imágenes
cat <<EOL > backend/app/services/image_detection.py
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
EOL

# Crear archivo de detección de vídeos
cat <<EOL > backend/app/services/video_detection.py
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
                "color": color_hex
            })

        frame_count += 1

    cap.release()
    return detected_faces
EOL

# Crear archivo de base de datos
cat <<EOL > backend/app/db/db.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
EOL

# Crear archivo de inicio
cat <<EOL > backend/start.sh
#!/bin/bash

uvicorn app.api.main:app --reload
EOL

chmod +x backend/start.sh

# Eliminar la vieja estructura si existe
rm -rf backend/app/api/pictures.py
rm -rf backend/app/api/main.py
rm -rf backend/app/api/schemas.py
rm -rf backend/app/models/models.py
rm -rf backend/app/crud/crud.py
rm -rf backend/app/utils/utils.py

echo "Estructura de proyecto creada exitosamente."
