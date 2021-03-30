# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import pandas as pd
import jieba.analyse
import sklearn.feature_extraction.text as ft
import sklearn.model_selection as ms
import sklearn.naive_bayes as nb
import sklearn.metrics as sm

# 加载文件
data = pd.read_csv('datasets/htl_all.csv')
# 加载自定义词典
jieba.load_userdict('datasets/mydic.txt')
sentences = []
for i, line in enumerate(data['review'].values):
    sentences.append(' '.join(jieba.cut(line, HMM=False)))

# 处理训练集样本类别均衡化
cls_num_0 = (data['label'] == 0).sum()
cls_num_1 = (data['label'] == 1).sum()
sentences, labels = \
    sentences[cls_num_1 - cls_num_0:], data['label'][cls_num_1 - cls_num_0:]

# 拆分测试集，训练集
train_x, test_x, train_y, test_y = \
    ms.train_test_split(sentences, labels,
                        test_size=0.1, random_state=7)

cv = ft.CountVectorizer()
train_bow = cv.fit_transform(train_x)
tt = ft.TfidfTransformer()
tfidf = tt.fit_transform(train_bow)
print(tfidf.shape)
train_x = tt.fit_transform(train_bow)
model = nb.MultinomialNB(fit_prior=True)
model.fit(train_x, train_y)

# 训练集分类报告
pred_train_y = model.predict(train_x)
print('train-data classification report:')
print(sm.classification_report(train_y, pred_train_y))
# 测试集分类报告
test_bow = cv.transform(test_x)
test_x = tt.transform(test_bow)
pred_test_y = model.predict(test_x)
print('test-data classification report:')
print(sm.classification_report(test_y, pred_test_y))

# 自定义测试集数据：
test_data = ['这房子隔音差的厉害，再也不会来这里住了。',
             '感觉还可以，可以接受，但不算太好。',
             '地板挺好不凉，电视比较给力，隔音不错。',
             '地板漏水，空调也坏，隔音也不行，真差劲。',
             '我不喜欢。', '不好', '不喜欢',
             '酒店有点远，大堂电脑都是坏的，地板还脏，但是感觉还行。']
sentences = []
for i, line in enumerate(test_data):
    sentences.append(' '.join(jieba.cut(line, HMM=False)))
test_bow = cv.transform(sentences)
test_x = tt.transform(test_bow)
pred_test_y = model.predict(test_x)
probs = model.predict_proba(test_x)

for sentence, index, prob in zip(
        test_data, pred_test_y, probs.max(axis=1)):
    print(sentence, '->', index, '->', prob)
