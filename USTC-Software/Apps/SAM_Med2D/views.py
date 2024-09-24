from django.shortcuts import render, HttpResponse

# Create your views here.

def sam_index(request):
    return HttpResponse('this is the sam index')


# 调用SAM_Med2D的模型
