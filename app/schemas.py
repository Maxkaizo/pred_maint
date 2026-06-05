from datetime import datetime

from pydantic import BaseModel


class TelemetryData(BaseModel):
    """Schema for raw telemetry sensor data."""
    datetime: datetime
    machineID: int
    volt: float
    rotate: float
    pressure: float
    vibration: float

class ErrorData(BaseModel):
    """Schema for machine error events."""
    datetime: datetime
    machineID: int
    errorID: str

class MaintenanceData(BaseModel):
    """Schema for machine maintenance events."""
    datetime: datetime
    machineID: int
    comp: str

class FailureData(BaseModel):
    """Schema for machine failure events."""
    datetime: datetime
    machineID: int
    failure: str
