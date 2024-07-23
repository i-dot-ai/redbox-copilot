from django.urls import path

from .redbox_core import consumers

websocket_urlpatterns = [
    path(r"ws/chat/", consumers.ChatConsumer.as_asgi(), name="chat"),
]
