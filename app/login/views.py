# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from login.models import Person
from django.views.decorators.csrf import ensure_csrf_cookie

# Create your views here.

@ensure_csrf_cookie
def events(request):
    #if request.method == "POST":
    print('in events')
    print(request.method)
    if request.method == "POST":
    	if 'name' in request.POST:
    		open(request.POST['recording'], 'wb')
    		print(request.POST['name'])
    		print(type(request.POST['recording']))
    		new_Person = Person(name=request.POST['name'], recording=request.POST['recording'])
    		new_Person.save()
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