# modules/data_fetching.py
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import pandas as pd
from config import DB_SERVER, DB_NAME, DB_PASSWORD, DB_USER, DB_PORT


odbc_connect_str = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER=tcp:{DB_SERVER},1433;"
    f"DATABASE={DB_NAME};"
    f"UID={DB_USER};"
    f"PWD={DB_PASSWORD};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=30;"
)

connection_url = URL.create(
    "mssql+pyodbc",
    query={"odbc_connect": odbc_connect_str}
)

engine = create_engine(connection_url)

def get_engine():
    return engine


def fetch_user_by_id(user_id):
    engine = get_engine()
    query = f"""
        SELECT Users.Id, Users.Name, Users.Email, Users.DateOfBirth, Users.DistanceWillingToTravel, 
               Locations.Latitude, Locations.Longitude
        FROM Users
        LEFT JOIN Locations ON Users.LocationId = Locations.Id
        WHERE Users.Id = {user_id}
    """
    return pd.read_sql(query, engine)

def fetch_users():
    engine = get_engine()
    return pd.read_sql("""
        SELECT Users.Id, Users.Name, Users.Email, Users.DateOfBirth, Users.DistanceWillingToTravel, 
               Locations.Latitude, Locations.Longitude
        FROM Users
        LEFT JOIN Locations ON Users.LocationId = Locations.Id
    """, engine)

def fetch_activities():
    engine = get_engine()
    return pd.read_sql("""
        SELECT Activities.Id, Activities.Name, Activities.Description, Activities.LocationId, Activities.Date, Activities.CreatedById, 
               Locations.Latitude, Locations.Longitude
        FROM Activities
        LEFT JOIN Locations ON Activities.LocationId = Locations.Id
    """, engine)

def fetch_locations():
    engine = get_engine()
    return pd.read_sql("SELECT Id, Latitude, Longitude FROM Locations", engine)

def fetch_subcategories():
    engine = get_engine()
    return pd.read_sql("SELECT Id, Name, CategoryId FROM Subcategories", engine)

def fetch_user_subcategories():
    engine = get_engine()
    return pd.read_sql("SELECT UserId, SubcategoryId, Weight FROM UserSubcategories", engine)

def fetch_user_feedbacks(user_id):
    engine = get_engine()
    feedbacks_query = f"""
        SELECT Rating, TargetUserId, Timestamp
        FROM Feedbacks
        WHERE UserId = {user_id}
        ORDER BY Timestamp DESC
    """
    return pd.read_sql(feedbacks_query, engine)

def fetch_activity_by_id(activity_id):
    """
    Fetch a single activity by its ID, including associated location data.
    Returns a DataFrame with the queried activity.
    """
    engine = get_engine()
    query = f"""
        SELECT A.Id, A.Name, A.Description, A.LocationId, A.Date, A.CreatedById,
               L.Latitude, L.Longitude
        FROM Activities AS A
        LEFT JOIN Locations AS L ON A.LocationId = L.Id
        WHERE A.Id = {activity_id}
    """
    return pd.read_sql(query, engine)
