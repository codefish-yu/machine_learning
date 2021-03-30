'''
电影推荐
数据样本ratings.json中记录着用户和他们看过的电影，
以键值对形式：{用户名1:{电影1:对该电影的评分,电影２:对该电影的评分...},用户名２:...}
'''

import json
import numpy as np

#打开数据样本文件（r:读方式，问加你必须存在)
with open('/home/tarena/fifthStage/machinelearning/myday15/ratings.json','r') as f:
    #json.loads:将json格式的字符串转为pyhton数据类型
    ratings = json.loads(f.read())
#print(ratings)

#查看数据样本中有多少个用户
users = list(ratings.keys())
# print(users)

#基于双层for循环，生成相似度矩阵
#创建相似度矩阵容器
scmat = []
for user1 in users:
    #创建容器存放user1对user2的相似度得分
    scrow  =[]
    for user2 in users:
        #计算user1与user2的相似度得分

        #创建一个集合存放user1和user2都看过的电影
        #使用集合的原因：集合的元素有唯一性
        movies = set()

        #将user1与user2都看过的电影放到set()中
        #拿出user1看过的电影
        for movie in ratings[user1].keys():
            if movie in ratings[user2].keys():
                movies.add(movie)
        
        #如果他们没有共同看过的电影，相似度记为０
        if len(movies) == 0:
            score = 0
        #计算两用户相似度
        else:
            #建立两个列表存放两用户的电影评分(特征向量)
            A,B = [],[]
            for movie in movies:
                A.append(ratings[user1][movie])
                B.append(ratings[user2][movie])

            #＃方案一：使用欧式距离分数(欧式距离分数＝1/(1+欧式距离)衡量相似度
            #＃将ＡＢ列表转为ndarray对象用于计算
            #A,B = np.array(A),np.array(B)
            #score = 1/(1+np.sqrt(np.sum((Ａ - Ｂ)**2)))

            #方案二：使用皮尔逊相关系数衡量相似度
            #np.corrcoef(A,B)的返回值为相似度矩阵，[0,1]取出Ａ相对于自己和Ｂ的相似度值,参数为列表
            score = np.corrcoef(A,B)[0,1]
            #相关系数越接近０月表示两组样本不相关，相关系数越接近于１越表示两组样本正相关
            #相关系数越接近于－１，两组样本负相关(越不相关)

        #将user1对user2的相似度分数放到scrow中
        scrow.append(score)
    scmat.append(scrow) #生成相似度矩阵

#将相似度容器转为ndarray数组：是个二位数组，每一行的数组代表该用户与其他用户的相似度
scmat = np.array(scmat)
print(np.round(scmat,2))    #四舍五入并保留两位小数

#以上推荐模型矩阵训练完毕
# 下边使用该模型做召回与排序

#按照推荐模型矩阵中的相似度从高到低,排列每个用户的相似用户
#遍历每个用户，模拟每个用户登录时的状态
#将users列表转为ndarray
users = np.array(users)
for i,user in enumerate(users): #enumerate()使遍历变成枚举遍历，不仅可以取出可迭代对象元素，还可取出对应的下标
    #1. 从scmat中取出该user与其他用户的相似度数组数据
    sorted_inds = scmat[i].argsort()[::-1]  #argsot()排序,返回值为下标，默认从小到大，使用[::-1]反转
    #将用户对自己的相似度得分的下标通过掩码的方式掩掉
    #numpy掩码：sorted_indices != i ，即Ｔrue的对应值会显示出来，false(= i)的下标会被掩掉
    sorted_inds = sorted_inds[sorted_inds != i]
    #拿出相似用户的人名
    sim_users = users[sorted_inds]
    #拿出相似用户的相似度数组数据
    sim_scores = scmat[i][sorted_inds]
    # print(user)
    # print(sim_users)
    # print(sim_scores)

    #找到所有正相关(皮尔逊相关系数>0）的相似用户
    #遍历每个相似用户，找到需要召回的电影并记录评分
    #使用字典来存放相似用户的电影和对应的评分：{'movie1':[1,2],'movie2':[1,2,3,2]...}

    #生成正向相似度样本掩码
    pos_mask = sim_scores >0
    #掩出所有正相关用户
    sim_users = sim_users[pos_mask]
    #创建一个存储推荐电影和评分的容器
    reco_movies = {}
    #遍历每个正相关用户找到需要召回的电影并存储电影名和相似用户评分
    for sim_user in sim_users:
        for movie in ratings[sim_user].keys():
            #如果当前user已看过，就不推荐了
            if movie not in ratings[user].keys():
                #如果user没看过该movie，取出相似用户对该电影的评分，放入reco_movies容器中
                score = ratings[sim_user][movie]
                if movie not in reco_movies.keys():
                    reco_movies[movie] = [score]
                else:
                    reco_movies[movie].append(score)

    # print(user)
    # print(reco_movies)

    #对reco_movies进行排序
    #sorted(iterable,key=None,reverse=False)
    #参数：key:主要用来进行比较的元素，来自于可迭代对象中，指定可迭代对象中的一个元素进行排序
    #参数：reverse:True降序，反之升序(默认)
    #dic.items():得到锁哦与键值对组成的列表
    #lambda表达式：lambda [参数１,参数２,..]:表达式
    sorted(reco_movies.items(),
           key=lambda x:np.mean(x[1]),  #x表示列表中的一个个元组，即对电影评分的均值进行排序
           reverse=True)










