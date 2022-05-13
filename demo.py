import json
import os
import tkinter.messagebox

import nltk
import torch
import jieba
from tkinter import *
import time

import params
from model import Decoder, Encoder, KnowledgeEncoder, Manager
from utils import Vocabulary, init_model, knowledgeToIndex
from search import ESChat, chat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置界面
class interface_init():
    def __init__(self, master, para):
        self.para = para
        self.master = master
        self.master.title('设置')
        Label(self.master, text="性别: ").grid(row=0)
        Label(self.master, text="地域: ").grid(row=1)
        Label(self.master, text="个性标签: ").grid(row=2)

        self.gender = Entry(self.master)
        self.region = Entry(self.master)
        self.personality = Entry(self.master)

        self.gender.grid(row=0, column=1, padx=10, pady=5)
        self.region.grid(row=1, column=1, padx=10, pady=5)
        self.personality.grid(row=2, column=1, padx=10, pady=5)

        Button(self.master, text="关闭", width=10, command=self.close).grid(row=4, column=0, sticky="w", padx=10, pady=5)
        Button(self.master, text="确定", width=10, command=self.init2dialog).grid(row=4, column=1, sticky="e", padx=10, pady=5)

        self.master.mainloop()

    def init2dialog(self):
        gender = self.gender.get()
        region = self.region.get()
        personality = self.personality.get()

        print(self.gender.get())
        print(self.region.get())
        print(self.personality.get())
        self.para.info = [gender, region, personality]
        self.para.K = [gender, region, personality]
        self.para.K = knowledgeToIndex(self.para.K, self.para.vocab)
        self.para.K = self.para.Kencoder(self.para.K)
        print()

        if gender == '' or region == '' or personality == '':
            tkinter.messagebox.showwarning('警告', '请完善个性化信息！')
        else:
            self.master.destroy()
            master_dialog = Tk()
            interface_dialog(master_dialog, self.para)

    def close(self):
        self.master.destroy()


# 对话界面
class interface_dialog():
    def __init__(self, master, para):
        self.para = para
        self.master = master
        self.num = 0

        self.master.title('聊天界面')
        # 创建分区
        self.frame_left_top = Frame(width=380, height=270, bg='white')  # 创建<消息列表分区 >
        self.frame_left_center = Frame(width=380, height=100, bg='white')  # 创建<发送消息分区 >
        self.frame_left_bottom = Frame(width=380, height=30)  # 创建<按钮分区>
        self.frame_right = Frame(width=170, height=400, bg='white')  # 创建<图片分区>
        # 创建控件
        self.text_msglist = Text(self.frame_left_top)
        self.text_msg = Text(self.frame_left_center)
        self.button_sendmsg = Button(self.frame_left_bottom, text='发送', command=self.dialog)
        self.button_back = Button(self.frame_left_bottom, text='返回', command=self.dialog2init)
        photo = PhotoImage(file='bot.png')
        label = Label(self.frame_right, image=photo)
        label.image = photo
        # 创建颜色tag
        self.text_msglist.tag_config('green', foreground='#008B00')
        self.text_msglist.tag_config('brown', foreground='#A52A2A')
        # 使用grid设置各个分区位置
        self.frame_left_top.grid(row=0, column=0, padx=10, pady=5)
        self.frame_left_center.grid(row=1, column=0, padx=10, pady=5)
        self.frame_left_bottom.grid(row=2, column=0)
        self.frame_right.grid(row=0, column=1, rowspan=3, padx=10, pady=5)
        self.frame_left_top.grid_propagate(0)
        self.frame_left_center.grid_propagate(0)
        self.frame_left_bottom.grid_propagate(0)
        # 把元素填充进frame
        self.text_msglist.grid()
        self.text_msg.grid()
        self.button_back.grid(row=0, column=0, sticky="e", padx=0)
        self.button_sendmsg.grid(row=0, column=1, sticky="w", padx=300)
        label.grid()

        self.master.mainloop()

    def dialog(self):
        self.num += 1

        msg = '我 ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n'
        self.text_msglist.insert(END, msg, 'green')  # 添加时间
        utterance = str(self.text_msg.get('0.0', END))
        self.text_msglist.insert(END, utterance)  # 获取发送消息，添加文本到消息列表
        self.text_msg.delete('0.0', END)  # 清空发送消息

        print('utterance =', utterance)

        if utterance == '':
            tkinter.messagebox.showwarning('警告', '不能发送空消息！')

        utterance_splice = utterance.replace(' ', '')
        utterance_jieba = jieba.cut(utterance_splice, cut_all=False)
        utterance_str = " ".join(utterance_jieba)
        if '性别' in utterance_splice:
            self.reply('我是' + self.para.info[0] + '生')
            return
        if '哪里' in utterance_splice:
            self.reply('我在' + self.para.info[1])
            return
        available_ES, answer = self.para.es_chat.chat(utterance_str)
        print('available_ES, answer =', available_ES, answer)
        if available_ES:
            self.reply(answer)
            return

        X = []
        tokens = nltk.word_tokenize(utterance_str)
        for word in tokens:
            if word in self.para.vocab.stoi:
                X.append(self.para.vocab.stoi[word])
            else:
                X.append(self.para.vocab.stoi["<UNK>"])
        X = torch.LongTensor(X).unsqueeze(0).to(device)  # X: [1, x_seq_len]

        encoder_outputs, hidden, x = self.para.encoder(X)
        k_i = self.para.manager(x, None, self.para.K)
        outputs = torch.zeros(self.para.max_len, 1, self.para.n_vocab).to(
            device
        )  # outputs: [max_len, 1, n_vocab]
        hidden = hidden[self.para.decoder.n_layer:]
        output = torch.LongTensor([params.SOS]).to(device)

        for t in range(self.para.max_len):
            output, hidden = self.para.decoder(output, k_i, hidden, encoder_outputs)
            outputs[t] = output
            output = output.data.max(1)[1]

        outputs = outputs.max(2)[1]

        answer = ""
        for idx in outputs:
            if idx == params.EOS:
                break
            answer += self.para.vocab.itos[idx][0]

        self.reply(answer)
        print("bot:", answer[:-1], "\n")

    def dialog2init(self):
        self.master.destroy()
        master_init = Tk()
        interface_init(master_init, self.para)

    def reply(self, answer):
        msg = 'bot ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n'
        self.text_msglist.insert(END, msg, 'brown')  # 添加时间
        self.text_msglist.insert(END, answer + '\n')  # 添加文本到消息列表

class parameters():
    def __init__(self):
        self.es_chat = ESChat(ip='localhost', port=9200, index_name='xiaohuangji')

        self.max_len = 50
        self.n_vocab = params.n_vocab
        self.n_layer = params.n_layer
        self.n_hidden = params.n_hidden
        self.n_embed = params.n_embed
        self.temperature = params.temperature

        if os.path.exists("vocab.json"):
            self.vocab = Vocabulary()
            with open("vocab.json", "r", encoding='utf-8') as f:
                self.vocab.stoi = json.load(f)

            for key in self.vocab.stoi.items():
                self.vocab.itos.append(key)
        else:
            print("vocabulary doesn't exist!")
            return

        print("loading model...")
        self.encoder = Encoder(self.n_vocab, self.n_embed, self.n_hidden, self.n_layer).to(device)
        self.Kencoder = KnowledgeEncoder(self.n_vocab, self.n_embed, self.n_hidden, self.n_layer).to(device)
        self.manager = Manager(self.n_hidden, self.n_vocab, self.temperature).to(device)
        self.decoder = Decoder(self.n_vocab, self.n_embed, self.n_hidden, self.n_layer).to(device)

        self.encoder = init_model(self.encoder, restore=params.encoder_restore)
        self.Kencoder = init_model(self.Kencoder, restore=params.Kencoder_restore)
        self.manager = init_model(self.manager, restore=params.manager_restore)
        self.decoder = init_model(self.decoder, restore=params.decoder_restore)
        print("successfully loaded!\n")

        self.K = []
        self.info = []  # 性别、地域、兴趣爱好标签

def main():
    para = parameters()
    root = Tk()
    interface_init(root, para)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
