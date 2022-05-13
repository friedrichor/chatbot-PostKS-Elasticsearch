# chatbot-PostKS-Elasticsearch
个性化聊天机器人系统

使用检索与生成混合模型，检索模块使用Elasticsearch+SBERT，生成模块使用Posterior-Knowledge-Selection

一、检索模块：Elasticsearch + SBERT  
1. Elasticsearch  
环境安装可以查看我写的一篇博客，使用的是8.1.2版本：[Elasticsearch环境搭建详细教程](https://blog.csdn.net/Friedrichor/article/details/124371742?spm=1001.2014.3001.5501)  
Elasticsearch文档：https://www.elastic.co/guide/cn/elasticsearch/guide/current/index.html  
2. Sentence BERT
论文：Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks  
论文下载网址：https://arxiv.org/abs/1908.10084  
使用sentence_transformers库：https://www.sbert.net/docs/quickstart.html#comparing-sentence-similarities  
可以参考：[NLP实践——基于SBERT的语义搜索，语义相似度计算，SimCSE、GenQ等无监督训练](https://blog.csdn.net/weixin_44826203/article/details/119868241?spm=1001.2014.3001.5506)  

二、生成模块：Posterior Knowledge Selection  
论文：Learning to Select Knowledge for Response Generation in Dialog Systems  
论文下载网址：https://arxiv.org/abs/1902.04911  
代码参考：https://github.com/bzantium/Posterior-Knowledge-Selection （github上的这个代码仅适用于英文对话，如果想要改为中文对话的训练，把其中使用glove模型的部分改为相应的中文模型即可，中文的word2vec等都可以）  

本项目仅仅是简单的实现，改进之处有很多，因此本项目仅供参考。
