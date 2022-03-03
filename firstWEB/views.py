from django.shortcuts import render
import draw as dw
import forecast as fc

def current(request):

    imd = dw.chart_current_time()

    data = dw.data_current()

    context = {
        'img': imd,
        'data': data,
    }
    return render(request,'current.html',context)

def provinces(request):

    imd1 = dw.chart_province()

    imd2 = dw.chart_province_time()

    imd3 = dw.chart_province_pollute()


    context = {
        'img1': imd1,
        'img2': imd2,
        'img3': imd3,

    }
    return render(request,'provinces.html',context)

def cities_search(request):

    imd = dw.chart_province()

    context = {
        "img": imd,
    }

    return render(request,'cities_search.html',context)

def cities_result(request):
    #获取搜索框的值
    city = request.POST['city']

    # 读取图片
    imd1 = dw.chart_city(city)
    imd2 = dw.chart_city_frequency(city)
    imd3 = dw.chart_city_time(city)
    imd4 = dw.chart_city_pollutePM2_5(city)
    imd5 = dw.chart_city_pollutePM10(city)
    imd6 = dw.chart_city_pollute(city)

    context = {
        'img1': imd1,
        'img2': imd2,
        'img3': imd3,
        'img4': imd4,
        'img5': imd5,
        'img6': imd6,
        'city': city,
    }
    return render(request,'cities_result.html',context=context)

def places_search(request):

    imd = dw.chart_city('济南市')

    context = {
        "img": imd,
    }

    return render(request,'places_search.html',context)

def places_result(request):
    # 获取搜索框的值
    place = request.POST['place']

    #imd1 = dw.chart_place_frequency(place)
    imd2 = dw.chart_place_time(place)
    imd3 = dw.chart_place_pollutePM2_5(place)
    imd4 = dw.chart_place_pollutePM10(place)
    imd5 = dw.chart_place_pollute(place)

    context = {
        #'img1': imd1,
        'img2': imd2,
        'img3': imd3,
        'img4': imd4,
        'img5': imd5,
        'place': place,

    }

    return render(request, 'places_result.html', context=context)

def analysis(request):
    return render(request,'analysis.html')

def forecast(request):

    imd = fc.get_forecast()

    context = {
        'img': imd,
    }
    return render(request,'forecast.html',context)
