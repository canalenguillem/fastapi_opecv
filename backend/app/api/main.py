from fastapi import FastAPI
from app.api.endpoints.pictures import router as pictures_router
from app.api.endpoints.videos import router as videos_router

app = FastAPI()

app.include_router(pictures_router, prefix="/pictures")
app.include_router(videos_router, prefix="/videos")

# Crear las tablas de la base de datos
from app.db.db import engine, Base
Base.metadata.create_all(bind=engine)
