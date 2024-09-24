"""EC URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""


from django.contrib import admin
from django.urls import path, include
from . import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('', views.index, name='home'), # 这是一个主页
    path("admin/", admin.site.urls),
    path('accounts/', include('Apps.accounts.urls', namespace='accounts')),
    path('forum/', include('Apps.forum.urls', namespace='forum')),
    path('chat_image/', include('Apps.chat_image.urls', namespace='chat_image')),
    path('raffle/', include('Apps.raffle.urls', namespace='raffle')),
    path('admin_panel/', include('Apps.admin_panel.urls', namespace='admin_panel')),
    
    # path('activities/', include('Apps.activities.urls', namespace='activities')),
    path('feedback/', include('Apps.feedback.urls', namespace='feedback')),
    # path('learning/', include('Apps.learning.urls', namespace='learning')),
    # path('mystery_hunt/', include('Apps.mystery_hunt.urls', namespace='mystery_hunt')),
    # path('settings/', include('Apps.settings.urls', namespace='settings')),
    # path('socialize/', include('Apps.socialize.urls', namespace='socialize')),
    path('works/', include('Apps.works.urls', namespace='works')),
    path('experiment/', include("Apps.experiment.urls", namespace='experiment')),
    path('image_segment/', include("Apps.SAM_Med2D.urls", namespace='image_segment')),
    
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# if settings.DEBUG:
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)