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
        color=face.color,
        path=face.path
    )
    db.add(db_face)
    db.commit()
    db.refresh(db_face)
    return db_face

def delete_video_and_faces(db: Session, video_id: int):
    video = db.query(Video).filter(Video.id == video_id).first()
    if video:
        # Eliminar archivos de rostros asociados al vídeo
        faces = db.query(VideoFace).filter(VideoFace.video_id == video_id).all()
        for face in faces:
            if os.path.exists(face.path):
                os.remove(face.path)
            db.delete(face)
        
        # Eliminar el archivo del vídeo
        if os.path.exists(video.path):
            os.remove(video.path)
        
        # Eliminar el vídeo de la base de datos
        db.delete(video)
        db.commit()
