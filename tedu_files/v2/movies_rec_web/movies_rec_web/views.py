from django.http import HttpResponse
from django.shortcuts import render
import requests
import json
import pandas as pd

def index(request):
    # loginid = {}
    # loginid['id'] = id
    return render(request, 'index.html')


#id从Url中传过来
def rec_list(request, id):
    url = 'http://127.0.0.1:8080/mr/common/userid/%s' % id

    #向推荐引擎发请求的爬虫代码
    headers = {'User-Agent': 'Mozilla/5.0'}
    # 用requests模块发请求,获取响应对象
    res = requests.get(url, headers=headers)
    # text属性 : 获取响应内容,json
    html = res.text
    print('-----------------1', html)
    #将json数据转为python数据类型
    data = json.loads(html)
    midlist = data['data']
    print('-----------------2', midlist)

    #整理一下要输出的数据的格式
    #pd.read_csv：pandas读取csv文件
    m_all_info = pd.read_csv('data/m_all_info.csv', header=None)
    #去除重复行
    m_all_info = m_all_info.drop_duplicates([0])
    #继续整理数据
    movieinfo = m_all_info[m_all_info[0].isin(midlist)].reset_index()
    data = []
    print(len(movieinfo))
    for j in range(len(movieinfo)):
        dic = {}
        for i in range(13):
            if i == 12:
                dic['f'+str(12)] = 'img/%s.jpg' % movieinfo[0][j]
            else:
                dic[i] = str(movieinfo[i][j])
        print(dic)

        data.append(dic)
    r = json.dumps(data)
    # for movie in data:
    #     print('movie_id:', movie[0], '  movie_name:', movie[1])

    return HttpResponse(r)  #将整理好的数据返回给前端