from django.db import models


class Database(models.Model):

    question = models.CharField(max_length=2000, blank=False)
    answer = models.CharField(max_length=2000, blank=False)
    interrogative = models.CharField(max_length=2000, blank=False)
    app_label = 'Database'

# Create your models here.
