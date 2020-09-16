from django.db import models

# Create your models here.

class Tweet(models.Model) :
    text = models.TextField(max_length=1024)
    sentiment = models.BinaryField()