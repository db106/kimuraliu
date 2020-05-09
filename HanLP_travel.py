# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:24:55 2020

@author: kimura
"""

from pyhanlp import *
import os, json
from time import time

#情感比對辭庫
path = r"D:\HanLP\test\ChnSentiCorp情感分析酒店评论"
plays = []
emotion={"正面":0,"负面":0}
NaiveBayesClassifier = JClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')

def predict(classifier, text):
    if classifier.classify(text) =="正面":
        emotion["正面"] = emotion["正面"] +1
    else:
        emotion["负面"] = emotion["负面"] +1
    print("《%s》 情感极性是 【%s】" % (text, classifier.classify(text)))
    #print("情感极性是 【{0}】".format(classifier.classify(text)))
    
#if __name__ == '__main__':

classifier = NaiveBayesClassifier()
#  创建分类器，更高级的功能请参考IClassifier的接口定义
classifier.train(path)
#  训练后的模型支持持久化，下次就不必训练了
#predict(classifier, "前台客房服务态度非常好！早餐很丰富，房价很干净。再接再厉！")
#predict(classifier, "结果大失所望，灯光昏暗，空间极其狭小，床垫质量恶劣，房间还伴着一股霉味。")
#predict(classifier, "可利用文本分类实现情感分析，效果不是不行")

#開啟景點json
f = open(r"D:\HanLP\text.json","r",encoding="utf-8")
plays = f.readlines()
f.close()
print(len(plays))

i=0
start = time()
for play in plays:
    playjson = json.loads(play)
    #文章內容分析
    
    print(i)
    i+=1
    predict(classifier,playjson["文章內容"])
    break
print(time() - start)