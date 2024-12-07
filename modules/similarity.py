# modules/similarity.py
from geopy.distance import geodesic
from datetime import datetime
import pandas as pd
import numpy as np

def calculate_distance(lat1, lon1, lat2, lon2):
    if None in [lat1, lon1, lat2, lon2]:
        return float('inf')
    return geodesic((lat1, lon1), (lat2, lon2)).km

def calculate_age(date_of_birth):
    if isinstance(date_of_birth, pd.Timestamp):
        date_of_birth = date_of_birth.to_pydatetime()
    elif isinstance(date_of_birth, np.datetime64):
        date_of_birth = pd.to_datetime(date_of_birth).to_pydatetime()
    today = datetime.today()
    return today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))
