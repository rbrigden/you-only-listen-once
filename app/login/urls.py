from django.conf.urls import include, url
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    url(r'^login$', views.login, name='login'),
    url(r'^register$', views.register, name="register"),
    url(r'^welcome/(?P<name>[\w\-]+)$', views.loggedIn, name="loggedIn"),
    url(r'^error$', views.error, name="error"),
]
