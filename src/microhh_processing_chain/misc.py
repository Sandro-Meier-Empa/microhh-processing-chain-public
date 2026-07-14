import datetime
from typing import Union

import numpy as np
import pandas as pd
from astropy import units
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time


def calculate_sun_position(
    lat: float,
    lon: float,
    time_utc: Union[datetime.datetime, np.datetime64, pd.Timestamp],
    height: float = 0,
) -> SkyCoord:
    """Calculate the Solar Zenith Angle (SZA) for a location at a specific time.

    Parameters:
    lat (float): Latitude of the location.
    lon (float): Longitude of the location.
    time_utc (Union[datetime.datetime, np.datetime64, pd.Timestamp]): UTC time.
    height (float, optional): Height of the location in meters. Default is 0.

    Returns:
    SkyCoord: The position of the sun in AltAz coordinates.
    """
    # Create an EarthLocation object for the given latitude and longitude
    location = EarthLocation(
        lat=lat * units.deg, lon=lon * units.deg, height=height * units.m
    )

    # Convert the UTC time to an Astropy Time object
    time = Time(time_utc, scale="utc")

    # Get the position of the sun at the given time and location
    altaz_frame = AltAz(obstime=time, location=location)
    sun_altaz = get_sun(time).transform_to(altaz_frame)

    return sun_altaz


__all__ = ["calculate_sun_position"]
