# Flat-Lattice-Transformer
code for ACL 2020 paper: FLAT: Chinese NER Using Flat-Lattice Transformer. 

Models and results can be found at our ACL 2020 paper [FLAT: Chinese NER Using Flat-Lattice Transformer](https://arxiv.org/pdf/2004.11795.pdf).




Requirement:
======
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

      Word(Lattice) embeddings (ctb.50d.vec): [Google Drive](https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)

2. Modify the `paths.py` to add the pretrained embedding and the dataset
3. Run following commands
```
cd V0
python preprocess.py (add '--clip_msra' if you need to train FLAT on MSRA NER dataset)
python flat_main.py --dataset <dataset_name> (ontonotes, msra, weibo or resume)
```
If you want to run the code on other datasets, you can write a function whose output format is like ''
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
