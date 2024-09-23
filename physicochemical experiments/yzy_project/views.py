from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import os
import shutil
import sys
from django.views.decorators.csrf import csrf_protect
import glob
from django.utils import translation
from django.shortcuts import redirect

def date_pro(request):
    return render(request, 'date_pro.html')

def experiments(request, experiment):
    experiment_title = experiment
    return render(request, 'experiment_process.html', {'title': experiment_title,
                                                        'success':'',
                                                        })

def explanation(request):
    return render(request, 'upload_explanation.html')

'''
def set_language(request):
    next_url = request.POST.get('next', '/')
    user_language = request.POST.get('language')
    translation.activate(user_language)
    response = redirect(next_url)
    response.set_cookie(settings.LANGUAGE_COOKIE_NAME, user_language)
    return response
'''
@csrf_protect 
def file_saved(request):
    if request.method == 'POST':
        files_remove = glob.glob(os.path.join(settings.MEDIA_ROOT, '*'))
        for file in files_remove:
            os.remove(file)
        download_files_remove = glob.glob(os.path.join(settings.MEDIA_ROOT,'*'))
        for file in download_files_remove:
            os.remove(file)
        remove_file_path = "download.zip"
        if os.path.exists(remove_file_path):
            os.remove(remove_file_path)
        print('post request')
        files = request.FILES.getlist('file')
        for file in files:
            if file:
                file_name = file.name
                file_path = os.path.join(settings.MEDIA_ROOT, file_name)
                with open(file_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
        experiment_title = request.POST.get('file_title')
        
        folder_start = settings.REFER_ROOT + '/' + experiment_title
        folder_final = [f for f in os.listdir(folder_start) if os.path.isfile(os.path.join(folder_start, f))]
        for file in folder_final:
            if not os.path.exists(os.path.join(settings.MEDIA_ROOT, file)):
                return render(request, 'experiment_process.html', {'title': experiment_title,
                                                                    'success':'未包含完整文件或文件名、拓展名与模版文件不符',
                                                                })
                                                                
        if experiment_title == "恒温槽的装配与性能测定":
            if 'experiment' not in sys.modules:
                from python_function import experiment
            experiment.process_txt_files()
        elif experiment_title == "分光光度法测BPB的电离平衡常数":
            if 'experimentTwo' not in sys.modules:
                from python_function import experimentTwo
            experimentTwo.xlsx_process()
        elif experiment_title == "燃烧热的测定":
            if 'experimentThree' not in sys.modules:
                from python_function import experimentThree
            experimentThree.process_three()
        elif experiment_title == "双液系的气液平衡相图":
            if 'experimentFour' not in sys.modules:
                from python_function import experimentFour
            experimentFour.date_process()
        elif experiment_title == "旋光物质化学反应反应动力学研究":
            if 'experimentFive' not in sys.modules:
                from python_function import experimentFive
            experimentFive.experimentFive_process()
        elif experiment_title == "乙酸乙酯皂化反应动力学研究":
            if 'experimentSix' not in sys.modules:
                from python_function import experimentSix
            experimentSix.experimentSix_process()
        elif experiment_title == "聚乙二醇的相变热分析":
            if 'experimentSeven' not in sys.modules:
                from python_function import experimentSeven
            experimentSeven.experimentSeven_process()
        elif experiment_title == "稀溶液粘度法测定聚合物的分子量":
            if 'experimentEight' not in sys.modules:
                from python_function import experimentEight
            experimentEight.experimentEight_process()
        for file in files:
            file_path = os.path.join(settings.MEDIA_ROOT, file.name)
            if os.path.exists(file_path):
                os.remove(file_path)
        return render(request, 'experiment_process.html', {'title': experiment_title,
                                                           'success':'数据处理成功',
                                                           })
    else:
        print('not a post request')
        # return render(request, 'experiment_process.html', {'title': experiment_title})
        return render(request, 'date_pro.html')
    

@csrf_protect   
def download(request):
    zip_file_name = 'download'
    shutil.make_archive(os.path.join(settings.UPLOAD_ROOT, zip_file_name), 'zip', settings.MEDIA_ROOT)
    zip_file_path = settings.UPLOAD_ROOT + '/{}.zip'.format(zip_file_name)
    with open(zip_file_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="{}.zip"'.format(zip_file_name)
        response['Content-Length'] = os.path.getsize(zip_file_path)
        return response