from django.conf.urls import include, url
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    url(r'^login$', views.events, name='events'),
    url(r'^register$', views.events1, name="events1"),
]
