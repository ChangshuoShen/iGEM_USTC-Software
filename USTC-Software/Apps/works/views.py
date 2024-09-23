from django.shortcuts import render

# Create your views here.
def jupyter(request):
    return render(request, 'jupyter.html')