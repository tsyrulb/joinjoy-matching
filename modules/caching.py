# modules/caching.py
import redis
import pickle
from config import REDIS_HOST, REDIS_PORT, REDIS_DB

cache = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def get_cached_profile(key, generate_func, *args, expiry=3600):
    if cache.exists(key):
        try:
            return pickle.loads(cache.get(key))
        except (pickle.UnpicklingError, UnicodeDecodeError):
            cache.delete(key)
    profile = generate_func(*args)
    cache.set(key, pickle.dumps(profile), ex=expiry)
    return profile
