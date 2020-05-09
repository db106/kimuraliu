# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:24:55 2020

@author: kimura

1.官網：http：//hanlp.com/

  pip3安裝pyhanlp

2，Python
  pip安裝pyhanlp
  
3.下載：data.zip（http://nlp.hankcs.com/download.php?file=data）
用戶可以自行增刪替換，如果不需要句法分析等功能的話，隨時可以刪除模型文件夾。
模型跟詞典沒有絕對的區別，隱馬模型被嵌入人人都可以編輯的詞典形式，不代表它不是模型。
GitHub代碼庫中已經包含了data.zip中的字典，直接編譯運行自動緩存即可；模型則需要額外下載。

4.下載jar和配置文件：hanlp-release.zip（http://nlp.hankcs.com/download.php?file=jar）

配置文件的作用是告訴HanLP數據包的位置，只需修改第一行
root = D：/ JavaProjects / HanLP /
為數據的父目錄即可，某些數據目錄是/ Users / hankcs / Documents / data，那麼root = / Users / hankcs / Documents /。
最後將hanlp.properties放入類路徑即可，對於多個項目，都可以放到src或資源目錄下，編譯時IDE會自動將其複製到類路徑中。
除了配置文件外，還可以使用環境變量HANLP_ROOT來設置root。安卓項目請參考demo。
如果放置不當，HanLP會提示當前環境下的合適路徑，並且嘗試從項目根目錄讀取數據集。

"""

from pyhanlp import *

import zipfile
import os
from pyhanlp.static import download, remove_file, HANLP_DATA_PATH

def test_data_path():
    """
    获取测试数据路径，位于$root/data/test，根目录由配置文件指定。
    :return:
    """
    HANLP_DATA_PATH = r"D:\HanLP"
    data_path = os.path.join(HANLP_DATA_PATH, 'test')
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    return data_path



## 验证是否存在语料库，如果没有自动下载
def ensure_data(data_name, data_url):
    root_path = test_data_path()
    dest_path = os.path.join(root_path, data_name)
    if os.path.exists(dest_path):
        print(dest_path)
        return dest_path
    
    if data_url.endswith('.zip'):
        dest_path += '.zip'
    download(data_url, dest_path)
    if data_url.endswith('.zip'):
        with zipfile.ZipFile(dest_path, "r") as archive:
            archive.extractall(root_path)
        remove_file(dest_path)
        dest_path = dest_path[:-len('.zip')]
    return dest_path


# 中文情感挖掘语料-ChnSentiCorp 谭松波
#下载 http://file.hankcs.com/corpus/ChnSentiCorp.zip 到 D:\HanLP\test\ChnSentiCorp情感分析酒店评论.zip
#100.00%, 1 MB, 437 KB/s, 还有 0 分  0 秒   秒 
chn_senti_corp = ensure_data("ChnSentiCorp情感分析酒店评论", "http://file.hankcs.com/corpus/ChnSentiCorp.zip")


## ===============================================
## 以下开始 情感分析


IClassifier = JClass('com.hankcs.hanlp.classification.classifiers.IClassifier')
NaiveBayesClassifier = JClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')



def predict(classifier, text):
    print("《%s》 情感极性是 【%s】" % (text, classifier.classify(text)))


if __name__ == '__main__':
    classifier = NaiveBayesClassifier()
    #  创建分类器，更高级的功能请参考IClassifier的接口定义
    classifier.train(chn_senti_corp)
    #  训练后的模型支持持久化，下次就不必训练了
    predict(classifier, "前台客房服务态度非常好！早餐很丰富，房价很干净。再接再厉！")
    predict(classifier, "结果大失所望，灯光昏暗，空间极其狭小，床垫质量恶劣，房间还伴着一股霉味。")
    predict(classifier, "可利用文本分类实现情感分析，效果不是不行")