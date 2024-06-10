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
