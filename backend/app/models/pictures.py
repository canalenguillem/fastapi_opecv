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
