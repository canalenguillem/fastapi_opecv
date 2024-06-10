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
