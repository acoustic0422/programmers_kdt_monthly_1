from django.shortcuts import render, HttpResponse
from django.views.defaults import page_not_found

# Create your views here.
def index(request):
    return render(request,'index.html')


def post(request):
    f = open('./showEDA/EDA_files/EDA_for_Monthly_Project.html', 'r', encoding='utf-8')
    test_text = f.read()
    return render(request, 'post.html', {"test": test_text})


def about(request):
    return render(request, 'about.html')


def test_markdown(request):
    test_text = "# this is the title\n this is text _italic_ and *bold*"
    # f = open('./showEDA/EDA_files/EDA_for_Monthly_Project.txt','r', encoding='utf-8')
    # test_text = f.read()

    return render(request, 'test.html', {"test": test_text})