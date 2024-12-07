# config.py
import os

# Database config
DB_SERVER = os.environ.get('DB_SERVER', 'localhost')
DB_NAME = os.environ.get('DB_NAME', 'join-joy-db')
TRUSTED_CONNECTION = 'yes'

# API endpoints
NET_CORE_API_BASE_URL = "https://localhost:7276/api/matching"
NET_CORE_API_BASE_URL_GEO = "https://localhost:7276/api/places"

# Redis config
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

# Model names/paths
SBERT_MODEL_NAME = 'paraphrase-MiniLM-L6-v2'

# Weight parameters (just moved from inline code)
KEY_WEIGHT = 0.3
VALUE_WEIGHT = 0.7
