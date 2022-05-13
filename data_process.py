import json
import os
import jieba
import sys
from tqdm import tqdm
import csv

from utils import build_vocab
import params

dataFile_search = 'data/xiaohuangji50w_nofenci.txt'
dataFile_personal = 'data/dialogues_train.json'
file_search = 'data/data_search.csv'
trainStdFile = 'data/train_dialogs_part.txt'  # 部分数据集的 10W
testStdFile= 'data/test_dialogs_part.txt'  # 部分数据集的 2W
trainDataFile = 'data/train_dialogs_part_final.txt'
testDataFile = 'data/test_dialogs_part_final.txt'
# trainStdFile = 'data/train_dialogs.txt'  # 整个数据集
# testStdFile = 'data/test_dialogs.txt'  # 整个数据集

def process_webtext2019():
    print('loading webtext2019')
    fr = open('data/webtext2019zh/web_text_zh_train.json', 'r', encoding='utf-8')
    fw = open('data/webtext_processed.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(fw)
    csv_writer.writerow(["question", "answer", ""])  # 构建列表头
    for line in tqdm(fr):
        question = json.loads(line).get("title")
        answer = json.loads(line).get("content")
        if question and answer:
            csv_writer.writerow([question, answer])
    fr.close()
    fw.close()



'''
def process_search(file_read, file_write):
    print('loading', file_read)
    fr = open(file_read, 'r', encoding='utf-8')
    fw = open(file_write, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(fw)
    csv_writer.writerow(["question", "answer", ""])  # 构建列表头

    # count = -1  # 计数，3n为E，3n+1为第一个句子，3n+2第2个句子
    dialog_pair = []  # 对话对
    for line in tqdm(fr):
        # count += 1
        line_split = line.split()
        if len(line_split) == 0:
            continue
        if line_split[0] == "E":
            dialog_pair = []
            continue
        if line_split[0] != "M" or len(line_split) < 2:
            continue
        # dialog_splice = line_split[1]
        # if dialog_splice == 'null':  # 去除句子是‘null’的
        #     continue
        dialog = line_split[1]
        if dialog == 'null':  # 去除句子是‘null’的
            continue
        # dialog_jieba = jieba.cut(dialog_splice, cut_all=False)
        # dialog = " ".join(dialog_jieba)
        dialog_pair.append(dialog)
        if len(dialog_pair) == 2:
            csv_writer.writerow(dialog_pair)
    fr.close()
    fw.close()
'''

def build_tag_vocab(dataFile):
    print('Build tag vocab!')
    vocab_tag = {}
    with open(dataFile, 'r', encoding='utf-8') as f:
        for line in f:
            profile = json.loads(line).get("profile")
            for character in range(2):  # 两个人物的对话
                tags = profile[character]['tag'][0]
                if len(tags) == 0:
                    continue
                tag_list = tags.split(';')
                for tag in tag_list:
                    if tag not in vocab_tag:
                        vocab_tag[tag] = 1
                    else:
                        vocab_tag[tag] += 1
    vocab_tag = dict((k, v) for k, v in vocab_tag.items() if v >= 1000)  # 取出高频tag
    vocab_tag = dict(sorted(vocab_tag.items(), key=lambda x: x[1], reverse=True))  # 按出现频次排序
    with open("vocab_tag.json", "w", encoding='utf-8') as fp:
        json.dump(vocab_tag, fp, ensure_ascii=False)
    print('vocab_tag.json has created!')



def loadData(dataFile, vocab_tag, trainFile, testFile):
    print('load', dataFile)
    file_train = open(trainFile, 'w', encoding='utf-8')
    with open(dataFile, 'r', encoding='utf-8') as f:
        num = 0
        for line in tqdm(f.readlines()):
            if num == 100000:
                break
            
            dialog_id = 1
            dialog_line_list = []
            # persona part
            profile_list = json.loads(line).get("profile")

            # gender & loc
            gender = profile_list[0]['gender']  # 只要第一个人物的gender信息
            loc = profile_list[0]['loc']  # 只要第一个人物的loc信息
            if len(gender) == 0 or len(loc) == 0 or loc == "其他":
                continue 
            dialog_std = str(dialog_id) + " your persona: " + ("男" if gender == "male" else "女") + "\n"
            dialog_line_list.append(dialog_std)
            dialog_id += 1
            loc_province = loc.split(' ')[0]  #  只要“省份”信息
            dialog_std = str(dialog_id) + " your persona: " + loc_province + "\n"
            dialog_line_list.append(dialog_std)
            dialog_id += 1

            # tag
            tag_new_list = []
            for character in range(2):  # 两个人物的对话
                tags = profile_list[character]['tag'][0]
                if len(tags) == 0:
                    continue
                tag_list = tags.split(';')
                # 取出高频tag
                for tag in tag_list:
                    if tag in vocab_tag:
                        tag_new_list.append(tag)
            tag_new_set = set(tag_new_list)  # set
            if len(tag_new_set) < 1:  # 高频tag不足1个，则不满足要求
                continue
            else:
                tag_new_list = list(tag_new_set)
            # tag 排序，并写入 your persona 部分
            tag_dict = {}
            for tag in tag_new_list:
                tag_dict[tag] = vocab_tag[tag]
            tags_sorted = sorted(tag_dict.items(), key=lambda x:x[1], reverse=True)
            tag_sorted_list = list(tag[0] for tag in tags_sorted)
            for tag in tag_sorted_list:
                tag_jieba = jieba.cut(tag, cut_all=False)
                dialog_std = str(dialog_id) + " your persona: " + " ".join(tag_jieba) + "\n"
                dialog_line_list.append(dialog_std)
                dialog_id += 1
            
            # dialog part
            dialog_list = json.loads(line).get("dialog")
            for i in range(1, len(dialog_list), 2):
                dialog_splice1 = dialog_list[i - 1][0].replace(' ', '')
                dialog_jieba1 = jieba.cut(dialog_splice1, cut_all=False)
                dialog_splice2 = dialog_list[i][0].replace(' ', '')
                dialog_jieba2 = jieba.cut(dialog_splice2, cut_all=False)
                dialog_std = str(dialog_id) + " " + " ".join(dialog_jieba1) + "\t" + " ".join(dialog_jieba2) + "\n"
                dialog_id += 1
                dialog_line_list.append(dialog_std)

            file_train.writelines(dialog_line_list)
            num += 1

    f.close()
    file_train.close()
    print('write', trainFile, 'successfully.')
    # with open(testFile, 'w', encoding='utf-8') as wf:
    #     for dialog in test_dialog_list:
    #         wf.writelines(dialog)
    #         wf.write('\n')
    # print('write', testFile, 'successfully.')

# 进一步处理数据，保证每个句子中的单词都在单词表中
def furtherLoadData(dataFile, vocab_tag, trainFile, testFile):
    print('load', dataFile)
    file_train = open(trainFile, 'w', encoding='utf-8')
    vocab = build_vocab(trainStdFile, params.n_vocab)
    keys_vocab = vocab.stoi.keys()
    with open(dataFile, 'r', encoding='utf-8') as f:
        num = 0
        for line in tqdm(f.readlines()):
            if num == 100000:
                break
            
            dialog_id = 1
            dialog_line_list = []
            # persona part
            profile_list = json.loads(line).get("profile")

            # gender & loc
            gender = profile_list[0]['gender']  # 只要第一个人物的gender信息
            loc = profile_list[0]['loc']  # 只要第一个人物的loc信息
            if len(gender) == 0 or len(loc) == 0 or loc == "其他":
                continue 
            dialog_std = str(dialog_id) + " your persona: " + ("男" if gender == "male" else "女") + "\n"
            dialog_line_list.append(dialog_std)
            dialog_id += 1
            loc_province = loc.split(' ')[0]  #  只要“省份”信息
            dialog_std = str(dialog_id) + " your persona: " + loc_province + "\n"
            dialog_line_list.append(dialog_std)
            dialog_id += 1

            # tag
            tag_new_list = []
            for character in range(2):  # 两个人物的对话
                tags = profile_list[character]['tag'][0]
                if len(tags) == 0:
                    continue
                tag_list = tags.split(';')
                # 取出高频tag
                for tag in tag_list:
                    if tag in vocab_tag:
                        tag_new_list.append(tag)
            tag_new_set = set(tag_new_list)  # set
            if len(tag_new_set) < 1:  # 高频tag不足1个，则不满足要求
                continue
            else:
                tag_new_list = list(tag_new_set)
            # tag 排序，并写入 your persona 部分
            tag_dict = {}
            for tag in tag_new_list:
                tag_dict[tag] = vocab_tag[tag]
            tags_sorted = sorted(tag_dict.items(), key=lambda x:x[1], reverse=True)
            tag_sorted_list = list(tag[0] for tag in tags_sorted)
            for tag in tag_sorted_list:
                tag_jieba = jieba.cut(tag, cut_all=False)
                dialog_std = str(dialog_id) + " your persona: " + " ".join(tag_jieba) + "\n"
                dialog_line_list.append(dialog_std)
                dialog_id += 1
            
            # dialog part
            dialog_list = json.loads(line).get("dialog")
            stop = False  # 句子中的单词是否都在词表中，stop表示终止提取对话
            for i in range(1, len(dialog_list), 2):
                dialog_splice1 = dialog_list[i - 1][0].replace(' ', '')
                dialog_jieba1 = jieba.cut(dialog_splice1, cut_all=False)
                dialog1 = " ".join(dialog_jieba1)
                dialog_splice2 = dialog_list[i][0].replace(' ', '')
                dialog_jieba2 = jieba.cut(dialog_splice2, cut_all=False)
                dialog2 = " ".join(dialog_jieba2)
                for word in dialog1.split():
                    if word not in keys_vocab:
                        stop = True
                        break
                if stop == True:
                    break
                for word in dialog2.split():
                    if word not in keys_vocab:
                        stop = True
                        break
                if stop == True:
                    break
                dialog_std = str(dialog_id) + " " + dialog1 + "\t" + dialog2 + "\n"
                dialog_id += 1
                dialog_line_list.append(dialog_std)

            if stop == False:
                file_train.writelines(dialog_line_list)
                num += 1
                
    f.close()
    file_train.close()
    print('write', trainFile, 'successfully.')
    # with open(testFile, 'w', encoding='utf-8') as wf:
    #     for dialog in test_dialog_list:
    #         wf.writelines(dialog)
    #         wf.write('\n')
    # print('write', testFile, 'successfully.')


if __name__ == "__main__":
    # search dialog
    # process_search(dataFile_search, file_search)  # 小黄鸡
    process_webtext2019()# webtext2019

    # personal dialog
    # build_tag_vocab(dataFile_personal)
    # with open('vocab_tag.json', 'r', encoding='utf-8') as fp:
    #     vocab_tag = json.load(fp)
    # loadData(dataFile_personal, vocab_tag, trainStdFile, testStdFile)
    # furtherLoadData(dataFile_personal, vocab_tag, trainDataFile, testDataFile)
