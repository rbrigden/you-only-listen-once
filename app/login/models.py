# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

from django.contrib.auth.models import User

# Create your models here.

class Person(models.Model):
	name = models.CharField(max_length = 255)
	recording = models.FileField(null = True, blank=True)

