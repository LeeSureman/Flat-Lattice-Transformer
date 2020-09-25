[English](#Requirement)
[中文](#运行环境)

# Flat-Lattice-Transformer
code for ACL 2020 paper: FLAT: Chinese NER Using Flat-Lattice Transformer. 

Models and results can be found at our ACL 2020 paper [FLAT: Chinese NER Using Flat-Lattice Transformer](https://arxiv.org/pdf/2004.11795.pdf).




# Requirement:

```
Python: 3.7.3
PyTorch: 1.2.0
FastNLP: 0.5.0
Numpy: 1.16.4
```
you can go [here](https://fastnlp.readthedocs.io/zh/latest/) to know more about FastNLP.



How to run the code?
====
1. Download the character embeddings and word embeddings.

      Character and Bigram embeddings (gigaword_chn.all.a2b.{'uni' or 'bi'}.ite50.vec) : [Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)

      Word(Lattice) embeddings: 
      
      yj, (ctb.50d.vec) : [Google Drive](https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)
      
      ls, (sgns.merge.word.bz2) : [Baidu Pan](https://pan.baidu.com/s/1luy-GlTdqqvJ3j-A4FcIOw)

2. Modify the `paths.py` to add the pretrained embedding and the dataset
3. Run following commands
```
python preprocess.py (add '--clip_msra' if you need to train FLAT on MSRA NER dataset)
cd V0 (without Bert) / V1 (with Bert)
python flat_main.py --dataset <dataset_name> (ontonotes, msra, weibo or resume)
```

If you want to record experiment result, you can use fitlog:
```
pip install fitlog
fitlog init V0
cd V0
fitlog log logs
```
then set use_fitlog = True in flat_main.py.

you can go [here](https://fitlog.readthedocs.io/zh/latest/) to know more about Fitlog.


Cite: 
========
[bibtex](https://www.aclweb.org/anthology/2020.acl-main.611.bib)

---






# 运行环境:

```
Python: 3.7.3
PyTorch: 1.2.0
FastNLP: 0.5.0
Numpy: 1.16.4
```
你可以在 [这里](https://fastnlp.readthedocs.io/zh/latest/) 深入了解 FastNLP 这个库.



如何运行？
====
1. 请下载预训练的embedding

      从[Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) 或 [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D) 下载字和 Bigram 的 embedding (gigaword_chn.all.a2b.{'uni' or 'bi'}.ite50.vec) 

      从[Google Drive](https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view?usp=sharing) 或 [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D) 下载词的 embedding (ctb.50d.vec)(yj)
      
      从[Baidu Pan](https://pan.baidu.com/s/1luy-GlTdqqvJ3j-A4FcIOw) 下载词的embedding (sgns.merge.bigram.bz2)(ls)

2. 修改 `paths.py` 来添加预训练的 embedding 和你的数据集
3. 运行下面的代码
```
python preprocess.py (add '--clip_msra' if you need to train FLAT on MSRA NER dataset)
cd V0 (without Bert) / V1 (with Bert)
python flat_main.py --dataset <dataset_name> (ontonotes, msra, weibo or resume)
```

如果你想方便地记录和观察实验结果, 你可以使用fitlog:
```
pip install fitlog
fitlog init V0
cd V0
fitlog log logs
```
然后把flat_main.py里的 use_fitlog 设置为 True 就行
你可以在 [这里](https://fitlog.readthedocs.io/zh/latest/) 深入了解 Fitlog 这个工具


引用: 
========
[bibtex](https://www.aclweb.org/anthology/2020.acl-main.611.bib)

