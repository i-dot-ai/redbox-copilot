from django.conf import settings


def compression_enabled(_request):
    """we want to turn compression off during testing"""
    return {"COMPRESSION_ENABLED": settings.COMPRESSION_ENABLED}
