# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_bow.py
"""
import nltk.tokenize as tk
import sklearn.feature_extraction.text as ft
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
