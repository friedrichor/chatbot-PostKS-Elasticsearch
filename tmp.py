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
    def __init__(self, master):
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

        if gender == '' or region == '' or personality == '':
            tkinter.messagebox.showwarning('警告', '请完善个性化信息！')
        else:
            self.master.destroy()
            master_dialog = Tk()
            interface_dialog(master_dialog)

    def close(self):
        self.master.destroy()


# 对话界面
class interface_dialog():
    def __init__(self, master):
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

        answer = "111"

        self.reply(answer)


    def dialog2init(self):
        self.master.destroy()
        master_init = Tk()
        interface_init(master_init)

    def reply(self, answer):
        msg = 'bot ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n'
        self.text_msglist.insert(END, msg, 'brown')  # 添加时间
        self.text_msglist.insert(END, answer + '\n')  # 添加文本到消息列表

def main():
    root = Tk()
    interface_init(root)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
