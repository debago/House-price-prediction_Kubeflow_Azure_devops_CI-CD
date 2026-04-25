from pydantic import BaseModel, Field


class HouseInput(BaseModel):
    MedInc: float = Field(..., gt=0)
    HouseAge: float = Field(..., ge=0)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., ge=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float = Field(..., ge=-90, le=90)
    Longitude: float = Field(..., ge=-180, le=180)