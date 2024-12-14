import redis
import pickle
from config import REDIS_URL

# Use REDIS_URL to connect directly if you prefer:
cache = redis.Redis.from_url(REDIS_URL)

def get_cached_profile(key, generate_func, *args, expiry=3600):
    if cache.exists(key):
        try:
            return pickle.loads(cache.get(key))
        except (pickle.UnpicklingError, UnicodeDecodeError):
            cache.delete(key)
    profile = generate_func(*args)
    cache.set(key, pickle.dumps(profile), ex=expiry)
    return profile
