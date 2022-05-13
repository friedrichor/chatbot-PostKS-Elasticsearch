import sys
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import jieba

class ESUtils(object):
    def __init__(self, index_name, create_index=False):
        self.es = Elasticsearch()
        self.index = index_name
        if create_index:
            mapping = {
                'properties': {
                    'question': {
                        'type': 'text',
                        'analyzer': 'ik_max_word',
                        'search_analyzer': 'ik_smart'
                    }
                }
            }
            # 创建index
            if self.es.indices.exists(index=self.index):
                self.es.indices.delete(index=self.index)
            self.es.indices.create(index=self.index)
            # 创建mapping
            self.es.indices.put_mapping(body=mapping, index=self.index)

    def insert_qa_pairs(self, qa_pairs, data_source):
        count = self.es.count(index=self.index)['count']  # 获取当前数据库中的已有document数量
        def gen_data():
            for i, qa in enumerate(qa_pairs):
                yield {
                    '_index': self.index,
                    '_id': i + count,
                    'data_source': data_source,
                    'question': qa[0],
                    'answer': qa[1]
                }
        bulk(self.es, gen_data())

class ESChat(object):
    def __init__(self, ip, port, index_name):
        self.es = Elasticsearch(hosts=ip, port=port)
        self.index = index_name
        self.model = SentenceTransformer('distiluse-base-multilingual-cased')

    def search(self, input_str):  # elasticsearch搜索得到候选语句集合
        """
        Args:
            input_str: 用户问句
        Returns: 由匹配的question、answer和其分数score构成的字典的列表
        """
        dsl = {
            "query": {
                "match": {
                    "question": input_str
                }
            }
        }
        hits = self.es.search(index=self.index, body=dsl)["hits"]["hits"]
        # print('search =', self.es.search(index=self.index, body=dsl))

        qa_pairs = []
        for h in hits:
            qa_pairs.append({'score': h['_score'], 'question': h['_source']['question'], 'answer': h['_source']['answer']})
        return qa_pairs

    def semantic_search(self, sentence, candidateList, topk=1):
        """
        :param sentence: 待匹配的语句
        :param candidateList: 候选语句集合
        :param topk: 返回前多少个答案
        :param model: 用于编码的模型
        :return [(dialog_similar, score)]: 结果和分数
        """
        topk = min(topk, len(candidateList))
        questions = []
        for qa_pair in candidateList:
            questions.append(qa_pair[0])
        sentence_emb = self.model.encode(sentence, convert_to_tensor=True)
        questions_emb = self.model.encode(questions, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(sentence_emb, questions_emb)[0]
        top_res = torch.topk(cosine_scores, k=topk)

        print('top_res =', top_res)
        result = [{'id': int(index.cpu()), 'q': candidateList[int(index.cpu())][0], 'a': candidateList[int(index.cpu())][1], 'score':float(score.cpu().numpy())}
                  for index, score in zip(top_res[1], top_res[0])]
        return result

    def chat(self, input_str):
        sentence = input_str
        qa_pairs = self.search(sentence)
        if not qa_pairs:  # 没有搜索到结果
            return False, '0'
        print('qa_pairs = ', qa_pairs)
        candidateList = []
        for qa_pair in qa_pairs:
            candidateList.append([qa_pair['question'], qa_pair['answer']])
        print(candidateList)
        result = self.semantic_search(input_str, candidateList, 10)
        print(result)
        if result[0]['score'] > 0.9:
            answer = result[0]['a']
            print(answer)
            return True, answer
        else:
            return False, ""


def get_qa_pairs(csv_path):
    qa_pairs = pd.read_csv(csv_path, error_bad_lines=False)
    qa_pairs = list(zip(qa_pairs['question'], qa_pairs['answer']))
    return qa_pairs

def chat(input_str):
    es_chat = ESChat(ip='localhost', port=9200, index_name='chatbot')
    available_ES, answer = es_chat.chat(input_str)
    return available_ES, answer

if __name__ == '__main__':
    # 小黄鸡训练
    # qa = get_qa_pairs("data/xiaohuangji_processed.csv")
    # es_util = ESUtils('xiaohuangji', True)
    # es_util.insert_qa_pairs(qa, 'data_source')
    # print('search part has finish!')

    # webtext2019训练
    # qa = get_qa_pairs("data/webtext_processed.csv")
    # es_util = ESUtils('webtext2019', True)
    # es_util.insert_qa_pairs(qa, 'data_source')
    # print('search part has finish!')

    # 测试
    input_str = 1
    es_chat = ESChat(ip='localhost', port=9200, index_name='xiaohuangji')
    while input_str != 0:
        input_str = input('> ')
        available_ES, answer = es_chat.chat(input_str)

