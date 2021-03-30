# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_tfidf.py  提取tfidf矩阵
"""
import nltk.tokenize as tk
import sklearn.feature_extraction.text as ft
import numpy as np
doc = 'The brown dog is running. ' \
      'The black dog is in the black room. ' \
      'Running in the room is forbidden.'
# print(doc)
sents = tk.sent_tokenize(doc)
print(sents)
# 提取词袋矩阵
cv = ft.CountVectorizer()
bow = cv.fit_transform(sents)  # 3个样本的list
print(bow.toarray())
# 获取词袋模型中的每个特征名
print(cv.get_feature_names())
# 提取tfidf  词频逆文档矩阵
tt = ft.TfidfTransformer()
tfidf = tt.fit_transform(bow)
print(np.round(tfidf.toarray(), 2))
