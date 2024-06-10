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
    path: str

class VideoFaceCreate(VideoFaceBase):
    pass

class VideoFace(VideoFaceBase):
    id: int
    class Config:
        orm_mode = True
