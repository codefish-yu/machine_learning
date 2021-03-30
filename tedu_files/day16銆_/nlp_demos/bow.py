# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import nltk.tokenize as tk
import sklearn.feature_extraction.text as ft
import jieba
import numpy as np
sentences = ['手机确实不错，外观设计优美，比宣传图片好看。',
             '手机不错，游戏画面感清晰，速度快。',
             '系统卡顿，差评，应用闪退，差评，屏幕没反应点不动，差评。']
cv = ft.CountVectorizer()
for i in range(len(sentences)):
    sentences[i] = ' '.join(jieba.cut(sentences[i]))
bow = cv.fit_transform(sentences)
print(bow.toarray())
words = cv.get_feature_names()
print(words)

tt = ft.TfidfTransformer()
tfidf = tt.fit_transform(bow)
print(np.round(tfidf.toarray(), 2))
