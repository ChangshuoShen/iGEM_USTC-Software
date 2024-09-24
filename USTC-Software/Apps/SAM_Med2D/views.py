from django.shortcuts import render, HttpResponse

# Create your views here.

def sam_index(request):
    return HttpResponse(request, 'this is the sam index')
