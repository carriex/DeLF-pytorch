
# Pytorch Implementation of DeLG
PyTorch Implementation of "Unifying Deep Local and Global Features for Image Search"      

reference: https://arxiv.org/pdf/2001.05027.pdf


## Prerequisites
+ PyTorch
+ python3
+ CUDA

## Training DeLG
After the train process is done, model will be saved at `repo/devel/finetune/ckpt`

~~~shell
$ cd train/
$ python main.py 
~~~

## Feature Extraction of DeLG

~~~shell
$ cd extraction/
$ python extractor.py 
~~~

## Evaluation 
~~~shell
$ python test.py
~~~
