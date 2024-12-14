import redis
import pickle
from config import REDIS_URL

cache = redis.Redis.from_url(REDIS_URL)

def get_cached_profile(key, generate_func, expiry=3600, **kwargs):
    if cache.exists(key):
        try:
            return pickle.loads(cache.get(key))
        except (pickle.UnpicklingError, UnicodeDecodeError):
            cache.delete(key)
    profile = generate_func(**kwargs)  
    cache.set(key, pickle.dumps(profile), ex=expiry)
    return profile

