import json
import csv
import pandas as pd
header = ['question', 'answer']
csvrow1 = []
csvrow2 = []
jsonPath=r"D:\chatbot_search\data\dialogues_train.json"
csvPath=r"D:\chatbot_search\data\dialogues_train.csv"
with open(jsonPath,'r',encoding='utf-8') as j:
    with open(csvPath,'a+',encoding='utf-8',newline='') as c:
        writer = csv.writer(c, delimiter=",")

        for jj in j:
            data = json.loads(jj)
            csvrow1.append(data['dialog'][0][0].replace('\n', '').replace('\r', ''))
            csvrow2.append(data['dialog'][1][0].replace('\n', '').replace('\r', ''))
        writer.writerow(header)
        writer.writerows(zip(csvrow1, csvrow2))


