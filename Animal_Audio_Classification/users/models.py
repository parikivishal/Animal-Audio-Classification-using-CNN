from django.db import models

# Create your models here.
class user_reg(models.Model):
    email = models.CharField(max_length=200)
    pwd = models.CharField(max_length=200)