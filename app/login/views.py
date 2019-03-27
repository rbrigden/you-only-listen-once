# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from login.models import Person
from django.views.decorators.csrf import ensure_csrf_cookie
import pyaudio
import wave
import io
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

#@ensure_csrf_cookie
@csrf_exempt
def events(request):
    #if request.method == "POST":
    print('in events')
    print(request.method)
    if request.method == "POST":
        print('in POST')
        #print(request.POST['name'])
        print(type(request.FILES['picture']))
        play_audio(request.FILES['picture'])
        #play_audio(bytes(request.POST['recording']))
        #print('after play blob')
        #new_Person = Person(name=request.POST['name'], recording=request.POST['recording'])
        #new_Person.save()
        return render(request, 'login/home.html', {})
    return render(request, 'login/home.html', {})


def events1(request):
    #if request.method == "POST":
    print('in events')
    print(request.method)
    if request.method == "POST":
        if 'name' in request.POST:
            print('hello')
            print(request.POST['name'])
            print(type(request.POST['recording']))
            new_Person = Person(name=request.POST['name'], recording=request.POST['recording'])
            new_Person.save()
            return render(request, 'login/home.html', {})
    return render(request, 'login/register.html', {})


def play_audio(blob):
    #define stream chunk                                                                                               
    chunk = 1024
    #print()
    #f = io.BytesIO(blob)

    #open a wav format music                                                                                           
    f = wave.open(blob, 'rb')
    #instantiate PyAudio                                                                                               
    p = pyaudio.PyAudio()
    #open stream                                                                                                       
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
                    channels = f.getnchannels(),
                    rate = f.getframerate(),
                    output = True)
    #read data                                                                                                         
    data = f.readframes(chunk)

    #play stream                                                                                                       
    while data:
        stream.write(data)
        data = f.readframes(chunk)

    #stop stream                                                                                                       
    stream.stop_stream()
    stream.close()

    #close PyAudio                                                                                                     
    p.terminate()
    return

