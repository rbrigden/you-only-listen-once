# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2019-03-17 18:38
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('login', '0003_person_recording'),
    ]

    operations = [
        migrations.AlterField(
            model_name='person',
            name='recording',
            field=models.FileField(blank=True, null=True, upload_to=b''),
        ),
    ]
