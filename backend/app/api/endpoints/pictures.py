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
