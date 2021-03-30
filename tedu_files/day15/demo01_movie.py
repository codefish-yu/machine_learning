# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_movie.py 电影推荐
"""
import json
import numpy as np

with open('../ml_data/ratings.json', 'r') as f:
    ratings = json.loads(f.read())
# print(ratings)
users = list(ratings.keys())
# 基于双层for循环，生成相似度矩阵
scmat = []
for user1 in users:
    scrow = []
    for user2 in users:
        # 计算user1与user2的相似度得分
        movies = set()
        for movie in ratings[user1].keys():
            if movie in ratings[user2].keys():
                movies.add(movie)
        if len(movies) == 0:
            score = 0
        else:
            A, B = [], []  # 保存两用户的特征向量
            for movie in movies:
                A.append(ratings[user1][movie])
                B.append(ratings[user2][movie])
            # 基于欧氏距离得分，计算相似度
            # A, B = np.array(A), np.array(B)
            # score = 1 / (
            #     1 + np.sqrt(np.sum((A - B)**2)))
            score = np.corrcoef(A, B)[0, 1]

        scrow.append(score)
    scmat.append(scrow)
scmat = np.array(scmat)
print(np.round(scmat, 2))
# 推荐模型矩阵训练完毕   下边使用该模型做召回与排序

# 遍历每一个用户，模拟每个用户登录时的状态
users = np.array(users)
for i, user in enumerate(users):
    # 获取每个用户的相似用户 排序
    sorted_inds = scmat[i].argsort()[::-1]
    sorted_inds = sorted_inds[sorted_inds != i]
    sim_users = users[sorted_inds]
    sim_scores = scmat[i][sorted_inds]
    # print(user)
    # print(sim_users)
    # print(sim_scores)
    # 找到所有正相关的相似用户
    pos_mask = sim_scores > 0
    sim_users = sim_users[pos_mask]
    # {'movie1':[1, 2], 'movie2':[1, 2, 3 ,2]...}
    reco_movies = {}
    for sim_user in sim_users:
        # 遍历每个相似用户，找到需要召回的电影并记录评分
        for movie in ratings[sim_user].keys():
            if movie not in ratings[user].keys():
                # movie就是一部召回的电影
                score = ratings[sim_user][movie]
                if movie not in reco_movies.keys():
                    reco_movies[movie] = [score]
                else:
                    reco_movies[movie].append(score)

    # 针对reco_movies进行排序
    movie_list = sorted(reco_movies.items(),
                        key=lambda x: np.mean(x[1]),
                        reverse=True)
    print(user)
    print(movie_list)
