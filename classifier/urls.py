from django.urls import path
from .views import  *
app_name = 'classifier'

urlpatterns = [
    path('predict/',Predict.as_view(),name='predict') ,
]