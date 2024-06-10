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
    path = Column(String)
    
    video = relationship("Video", back_populates="faces")
