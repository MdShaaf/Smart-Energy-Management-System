#inputs
from pydantic import BaseModel, Field
class DemandInput(BaseModel):
    household_size: int = Field(default=4, ge=1,le=6, description="House Size 1,2,3,4,5 & 6, choose 6 for above")
    avg_temperature: float = Field(default=30, ge=-10, le= 60, description="Temperature in Celsius.")
    has_ac: int = Field (default=0, ge=0, le=1,description= "Yes : 1, No:0")
    avg_humidity: float = Field (default=70, ge=50, le=100,description="Average humidity in(%).")
    day: int = Field(default=1, ge=1, le=31,description="Day of the month (1-31).")
    month: int  = Field(default=1, ge=1, le=12,description="Month of the year (1-12).")
    is_weekend: int = Field(default=0, ge=0, le=1,description=" Weekend : 1, Weekday : 0")
    is_holiday: int = Field(default=0, ge=0,le=1,description=" Holiday : 1, Workingday : 0")

class RainfallInput(BaseModel):
    temperature_2m : float = Field(default=30, ge=-10, le= 60,description="Temperature 2 meters above ground in Celsius.")
    relative_humidity_2m : float = Field (default=70,ge=50, le=100, description="Relative humidity 2 meters above ground (%).")
    cloud_cover : float = Field (default=50,ge=0, le=100,description="Percentage of cloud cover (0.0 to 100.0). Suggestion: Use recent satellite data.")
    pressure_msl : float = Field(default=1013.25, description="Atmospheric pressure at mean sea level (hPa). Standard is 1013.25.")
    wind_speed_10m :float = Field(default=12, ge=0, le=30,description="Wind speed 10 meters above ground in km/h or m/s.")
    month : int = Field(default=1, ge=1, le=12,description="Month of the year (1-12).")
    day : int = Field(default=1,ge=1, le=31, description="Day of the month (1-31).")
    Season : int = Field(default=1, ge=1, le=4, description="Winter: 1, Spring: 2, Summer: 3, Fall: 4")
