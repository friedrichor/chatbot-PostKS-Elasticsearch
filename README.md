# chatbot-PostKS-Elasticsearch
个性化聊天机器人系统

使用检索与生成混合模型，检索模块使用Elasticsearch+SBERT，生成模块使用Posterior-Knowledge-Selection

本项目仅仅是简单的实现，不足之处有很多，因此本项目仅供参考。

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

其中，word2vec里内容如下所示，因为大于100M了无法从GitHub Desktop上传，去网上找下中文word2vec词向量下载一下就行，这里用glove等等都是可以的，只要代码中路径改一下就行。  
![BIKO}VH}N)YY6J~JQ3P%_QB](https://user-images.githubusercontent.com/70964199/168311730-257672e6-7c08-4146-a337-9f1b9bd6448b.png)

snapshots中的内容如下所示，同样由于内存过大无法上传，这里存的是训练完的参数，训练后这部分会自动生成。  
![UN6CV2~H~%KQNJJTJ 6CEZN](https://user-images.githubusercontent.com/70964199/168313949-53d618ff-2292-4308-8a45-72e834b4f8d6.png)  

关于运行，搭建好Elasticsearch环境后执行search.py就配置好检索模块了，生成模型运行train.py，演示就执行demo.py，前端是用Tkinter写的。
