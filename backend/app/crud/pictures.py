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
