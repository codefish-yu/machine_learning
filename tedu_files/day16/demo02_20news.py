# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_20news.py
"""
import numpy as np
import sklearn.datasets as sd
import sklearn.naive_bayes as nb
import sklearn.feature_extraction.text as ft
import sklearn.model_selection as ms
import sklearn.metrics as sm

# 读取文档数据
data = sd.load_files(
    '../ml_data/20news', encoding='latin1',
    shuffle=True, random_state=7)
print(np.array(data.data)[0])
print(np.array(data.target)[0])
print(data.target_names)
# 通过样本数据，构建tfidf，训练分类模型
cv = ft.CountVectorizer()
bow = cv.fit_transform(data.data)
tt = ft.TfidfTransformer()
tfidf = tt.fit_transform(bow)
print(tfidf.shape)
# 拆分数据集
train_x, test_x, train_y, test_y = \
    ms.train_test_split(
        tfidf, data.target, test_size=0.1,
        random_state=7)
model = nb.MultinomialNB()
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.classification_report(test_y, pred_test_y))

# 自定义测试数据，预测类别
test_data = [
    'The curveballs of right handed pitchers tend to curve to the left',
    'Caesar cipher is an ancient form of encryption',
    'This two-wheeler is really good on slippery roads',
    "Visit refugee camps, pursue Muslim female knights, escape under the gun of the Taliban, ride a motorcycle alone to the most dangerous region in the world - the Middle East! Through Pakistan, Afghanistan, Iran, through life and death, through the whole energy Yes, what I'm going to talk about is a journey of adventure towards death and life. After leaving the post, the sun had set, and the Karakoram road soon fell into the dark. It was an hour's drive from the port of Suster, so I had to ride all night. The mountain wind whimpered, and there was no spark in the distance. There have been many robberies in this section of the road before. I'm a little nervous. I'm afraid that there will be several masked bandits with guns robbing chrysanthemum on the side of the road."]
bow = cv.transform(test_data)
tfidf = tt.transform(bow)
pred_code = model.predict(tfidf)
print(pred_code)
print(np.array(data.target_names)[pred_code])
# 置信概率
probs = model.predict_proba(tfidf)
print(np.round(probs, 3))
