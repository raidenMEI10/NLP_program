import re

import torch
from torch.utils.data import Dataset


def read_file_continuously(file_path, index):
    try:
        # 打开文件，以只读模式打开
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.readline()
            for i in range(index):
                text = file.readline()

            return text.strip()

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_string(s, n_number):
    # 移除第一个出现的数字及其后面的空格
    s = re.sub(r'\d*', '', s, count=len(str(n_number)))
    # s = re.sub(r'\s*', '', s, count=1)
    # result_list = re.split(r'(?<!\d)\. (?!\d)', s)

    # return result_list
    return s
class BIGDataset(Dataset):
    def __init__(self, corpus_path, text_len=2666763):
        super().__init__()
        self.corpus_path = corpus_path
        self.text_len = text_len

    def __getitem__(self, index):
        text = read_file_continuously(self.corpus_path, index)
        text = process_string(text, index)
        return text

    def __len__(self):
        return self.text_len


if __name__ == '__main__':
    dataset = BIGDataset('./doc.tsv')
    print(process_string(read_file_continuously('./doc.tsv', 0),0))
