# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import jieba
import jieba.analyse as ja

with open('datasets/zhurou.txt', 'rb') as f:
    jieba.load_userdict('dic.txt')
    doc = f.read()
    tags = ja.extract_tags(doc, topK=20, withWeight=False, allowPOS=())
    print('  '.join(tags))
    tags = ja.textrank(doc, topK=20, withWeight=False,
                       allowPOS=('ns', 'n', 'vn', 'v'))
    print('  '.join(tags))
