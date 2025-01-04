import re

import torch
from torch.utils.data import Dataset, DataLoader


def read_file_continuously(file_path, index):
    try:
        all_text = []
        # 打开文件，以只读模式打开
        with open(file_path, 'r', encoding='utf-8') as file:
            for i in range(index):
                text = file.readline()
                all_text.append(text.strip())

            return all_text

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def process_string(s, n_number):
    # 移除第一个出现的数字及其后面的空格
    s = re.sub(r'\d*', '', s, count=len(str(n_number)))
    s = re.sub(r'\s*', '', s, count=1)

    # 在第一个问号处切割字符串
    # split_str = re.split('？|?', s)
    split_str = s.split('?', 1)
    if len(split_str) ==1:
        split_str = s.split('？', 1)
    split_str[0] = split_str[0] + '?'
    match = re.search(r'\t+\d', split_str[1])
    if match:
        # 获取匹配结束的位置
        start_index = match.end()
        # 返回从匹配结束位置到字符串末尾的部分
        temp = split_str[1][1:start_index-2]
    match = re.search(r'\d+\t', split_str[1])
    if match:
        # 获取匹配结束的位置
        start_index = match.end()
        # 返回从匹配结束位置到字符串末尾的部分
        split_str[1] = split_str[1][start_index:]
    split_str.append(temp)


    # 返回处理后的结果
    return split_str

class QADataset(Dataset):
    def __init__(self, corpus_path, text_len=3079):
        super().__init__()
        self.corpus_path = corpus_path
        self.text_len = text_len
        self.corpus = read_file_continuously(self.corpus_path,self.text_len)

    def __getitem__(self, index):

        text = process_string(self.corpus[index], index)
        return text #text[0]为问题， text[1]为依据, text[3]为正确或错误

    def __len__(self):
        return self.text_len


if __name__ == '__main__':
    mystring = read_file_continuously('./test.tsv', 1) #第29行有中文问号
    print(mystring)
    print(process_string(mystring, 1))
    # dataset = QADataset('/test.tsv')
    # train_loader = DataLoader(dataset=dataset)
    # print(train_loader[0])
