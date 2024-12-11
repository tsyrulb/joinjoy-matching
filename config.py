# config.py
import os

DB_USER = "sa"
DB_PASSWORD = "YourStrongPassword123"
DB_SERVER = "db"
DB_NAME = "join-joy-db"
TRUSTED_CONNECTION = os.environ.get('TRUSTED_CONNECTION', 'no')  # Typically 'no' if using SQL auth

# .NET API endpoints
NET_CORE_API_BASE_URL = os.environ.get('NET_CORE_API_BASE_URL', 'http://webapi:5000/api/matching')
NET_CORE_API_BASE_URL_GEO = os.environ.get('NET_CORE_API_BASE_URL_GEO', 'http://webapi:5000/api/places')

# Redis config
REDIS_HOST = os.environ.get('REDIS_HOST', 'redis')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
REDIS_DB = int(os.environ.get('REDIS_DB', '0'))

# Model names/paths
SBERT_MODEL_NAME = os.environ.get('SBERT_MODEL_NAME', 'paraphrase-MiniLM-L6-v2')

# Weight parameters
KEY_WEIGHT = float(os.environ.get('KEY_WEIGHT', '0.3'))
VALUE_WEIGHT = float(os.environ.get('VALUE_WEIGHT', '0.7'))
