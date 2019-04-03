# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from login.models import Person
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.csrf import csrf_exempt
import redis
import hashlib
from datetime import datetime
import json
import asyncio

_redis_conn = None

def get_redis_conn():
    global _redis_conn
    if _redis_conn is None:
        _redis_conn = redis.Redis(host='localhost', port=6379)
    return _redis_conn


async def _fetch(key, conn):
    val = conn.get(key)
    while val is None:
        asyncio.sleep(1)
        val = conn.get(key)
    return val


def myconverter(o):
    if isinstance(o, datetime):
        return o.__str__()

def get_unique_id(audio_data):
    return hash_blob(audio_data)

def hash_blob(blob):
    md5 = hashlib.md5()

    # hash blob data
    md5.update(blob)

    # add timestamp to the hash
    md5.update(str(datetime.now()).encode('utf-8'))

    return md5.hexdigest()

# Create your views here.
# TODO: Will need CSRF for prod
@csrf_exempt
def events(request):
    print('in events')
    print(request.method)
    if request.method == "POST":
        print('POST request made')
        conn = get_redis_conn()
        audio_bytes = request.FILES['picture'].read()

        redis_request = {
            "id": get_unique_id(audio_bytes),
            "timestamp": datetime.now(),
            "type": "authenticate"
        }

        conn.rpush('queue:requests', json.dumps(redis_request, default=myconverter))
        conn.set('audio:{}'.format(redis_request['id']), audio_bytes)
        result = _fetch('result:{}'.format(redis_request['id']), conn)
        print('POST request serviced')
        return render(request, 'login/home.html', {})
    print('GET request made')
    return render(request, 'login/home.html', {})

@csrf_exempt
def events1(request):
    if request.method == "POST":
        print("Begin register")

        conn = get_redis_conn()
        audio_bytes = request.FILES['picture'].read()

        redis_request = {
            "id": get_unique_id(audio_bytes),
            "name": request.POST.get('name'),
            "timestamp": datetime.now(),
            "type": "register"
        }

        conn.rpush('queue:requests', json.dumps(redis_request, default=myconverter))
        conn.set('audio:{}'.format(redis_request['id']), audio_bytes)

        print("Complete register")
        return render(request, 'login/home.html', {})
    return render(request, 'login/register.html', {})


