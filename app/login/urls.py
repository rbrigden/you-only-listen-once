from django.conf.urls import include, url
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    url(r'^login$', views.events, name='events'),
    url(r'^register$', views.events1, name="events1"),
    url(r'^welcome/(?P<name>[\w\-]+)$', views.loggedIn, name="loggedIn"),
    url(r'^error$', views.error, name="error"),
]
