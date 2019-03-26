import gin
import redis
import json
import time
import os
import time
import sys
import hashlib
from datetime import datetime
import random

def hash_blob(blob):
    md5 = hashlib.md5()

    # hash blob data
    md5.update(blob)

    # add timestamp to the hash
    md5.update(str(datetime.now()).encode('utf-8'))

    return md5.hexdigest()

def myconverter(o):
    if isinstance(o, datetime):
        return o.__str__()

def get_unique_id(audio_data):
    return hash_blob(audio_data)

def build_fixtures():
    audio_fixtures_path = "app/processor/audio_fixtures"
    audio_paths = [os.path.join(audio_fixtures_path, file_name) for file_name in os.listdir(audio_fixtures_path)]
    requests = []
    audio_data = []
    for audio_path in audio_paths:
        f = open(audio_path, 'rb')
        audio = f.read()
        request = {
            "id": get_unique_id(audio),
            "timestamp": datetime.now()
        }
        requests.append(request)
        audio_data.append(audio)
        f.close()

    return zip(requests, audio_data)


def main_loop():
    conn = redis.Redis()

    fixtures = build_fixtures()

    for request, audio in fixtures:
        conn.rpush('queue:requests', json.dumps(request, default=myconverter))
        conn.set('audio:{}'.format(request['id']), audio)
        print("Web App pushed {}".format(request['id']))
        time.sleep(random.randint(1, 4))


if __name__ == "__main__":
    gin.external_configurable(redis.Redis, module="redis")
    gin.parse_config_file("app/processor/config/prod.gin")
    main_loop()

