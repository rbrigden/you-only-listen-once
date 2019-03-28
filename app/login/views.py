# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from login.models import Person
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

#@ensure_csrf_cookie
@csrf_exempt
def events(request):

    print('in events')
    print(request.method)
    if request.method == "POST":
        print(request.POST['name'])
        new_Person = Person(name=request.POST['name'], recording=request.POST['recording'])
        new_Person.save()
        return render(request, 'login/home.html', {})
    return render(request, 'login/home.html', {})


def events1(request):

    print('in events')
    print(request.method)
    if request.method == "POST":
        play_audio(request.FILES['picture'])
        print(request.POST['name'])
        new_Person = Person(name=request.POST['name'], recording=request.POST['recording'])
        new_Person.save()
        return render(request, 'login/home.html', {})
    return render(request, 'login/home.html', {})


