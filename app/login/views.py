# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from login.models import Person
from django.views.decorators.csrf import csrf_exempt
import asyncio
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.shortcuts import redirect
import redis
import hashlib
from datetime import datetime
import json
import time

_redis_conn = None

def get_redis_conn():
   global _redis_conn
   if _redis_conn is None:
       _redis_conn = redis.Redis(host='localhost', port=6379)
   return _redis_conn


def _fetch(key, conn):
    val = conn.get(key)
    while val is None:
        time.sleep(0.1)
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
@csrf_exempt
def login(request):
    User.objects.all().delete()
    print('in events')
    print(request.method)
    if request.method == "POST":
        text = request.POST['text']

        conn = get_redis_conn()
        audio_bytes = request.FILES['picture'].read()

        redis_request = {
            "id": get_unique_id(audio_bytes),
            "timestamp": datetime.now(),
            "type": "authenticate",
            "prompt": text
        }

        conn.set('audio:{}'.format(redis_request['id']), audio_bytes)
        time.sleep(0.1)
        conn.rpush('queue:requests', json.dumps(redis_request, default=myconverter))

        # Wait for the result
        result = _fetch('result:{}'.format(redis_request['id']), conn)
        result = json.loads(result.decode('utf-8'))


        print(result)
        if result['username']:
            user = User.objects.create_user(result['username'].strip(), 'user@gmail.com', 'yolorrn')
            user.save()
        else:
            result['username'] = 'None'

        response = {
            "username": result["username"].strip()
        }

        return HttpResponse(json.dumps(response), content_type='application/json')

    print('GET request made')
    return render(request, 'login/home.html', {})


@csrf_exempt
def loggedIn(request, name):
        #print(name)
        #json_context = '{ "username": "'+ name + '" }'
        #return HttpResponse(json_context, content_type='application/json')

        user = authenticate(request, username=name, password='yolorrn')
        if user is not None:
            return render(request, 'login/welcome.html')

@csrf_exempt
def error(request):
    return render(request, 'login/error.html')


@csrf_exempt
def register(request):
    if request.method == "POST":
        if 'name' in request.POST:
            name = request.POST['name']
            if not Person.objects.filter(username=name).exists():
                person = Person(username=name)
                person.save()
            else:
                message = 'Username already exists. Try again with valid name!'
                json_error = '{ "error": "'+message+'" }'
                return HttpResponse(json_error, content_type='application/json')

        conn = get_redis_conn()
        audio_bytes = request.FILES['picture'].read()

        redis_request = {
            "id": get_unique_id(audio_bytes),
            "name": request.POST.get('name'),
            "timestamp": datetime.now(),
            "type": "register"
        }

        conn.set('audio:{}'.format(redis_request['id']), audio_bytes)
        conn.rpush('queue:requests', json.dumps(redis_request, default=myconverter))
        return render(request, 'login/home.html', {})
        
    return render(request, 'login/register.html', {})


