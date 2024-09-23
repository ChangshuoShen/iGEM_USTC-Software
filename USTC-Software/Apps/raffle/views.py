from django.shortcuts import render, HttpResponse

# Create your views here.

def draw(request):
    return HttpResponse("Draw Prize")