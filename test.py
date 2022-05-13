from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('distiluse-base-multilingual-cased')

sentence1 = '呵呵'
sentence2 = '你好'
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)

# 计算语义相似度
cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
print(cosine_score)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction