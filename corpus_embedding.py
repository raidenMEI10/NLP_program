import re

import torch
from sentence_transformers import SentenceTransformer,util
from tqdm import tqdm

from Datas.QA_dataset import QADataset
from Datas.big_dataset import BIGDataset

def process_string(s, n_number):
    # 移除第一个出现的数字及其后面的空格
    s = re.sub(r'\d*', '', s, count=len(str(n_number)))
    s = re.sub(r'\s*', '', s, count=1)
    result_list = re.split(r'(?<!\d)\. (?!\d)', s)

    return result_list
def read_file_continuously(file_path,index):
    alltext = []
    try:
        # 打开文件，以只读模式打开
        with open(file_path, 'r', encoding='utf-8') as file:
            for i in range(index):
                text = file.readline()
                text = text.strip()
                text = process_string(text, index)
                alltext.extend(text)

        return alltext

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def getF1_score(prediction, ground_truth):
    pred_tokens = prediction.split()
    gt_tokens = ground_truth.split()

    # 创建集合以消除重复
    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)

    # 计算true positives, false positives, and false negatives
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    # 如果没有正样本或预测样本
    if tp + fp == 0 or tp + fn == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def eval(model, corpus_embeddings, test_loader,corpus):
    score = 0
    questions = []
    answers = []
    eval_loop = tqdm(test_loader)
    for query in test_loader:
        questions.append(query[0])
        answers.append(query[1])

        # query_embedding = model.encode(question, convert_to_tensor=True).to('cuda')
        #
        # similarity_scores = model.similarity(query_embedding, corpus_embeddings)[0]
        # scores, indices = torch.topk(similarity_scores, k=1)
        # result = indices[0]
        #
        #
        # f1 = getF1_score(result, answer)
        # score += f1/len(test_loader)
    query_embeddings = model.encode(questions)
    hits = util.semantic_search(query_embeddings, corpus_embeddings)
    for i in range(hits):
        result = corpus(hits[i]['corpus_id'])
        f1 = getF1_score(result, answers[i])
        score += f1/len(test_loader)



    return score




if __name__ == '__main__':

    model = SentenceTransformer("./all-MiniLM-L6-v2", device='cuda')
    print("Max Sequence Length:", model.max_seq_length)


    test_dataset = QADataset('./Datas/test.tsv')
    # corpus_loader = torch.utils.data.DataLoader(corpus_dataset,batch_size = 32)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1)
    # corpus_loop = tqdm(corpus_loader, desc='Train')
    print("begining to read files...")
    all_corpus = read_file_continuously('./Datas/doc.tsv', 2666763)
    print("len of corpus is "+str(len(all_corpus)))
    print("read is over, next is embedding")

    embeddings = model.encode(all_corpus, device='cuda').to("cuda")
    print(embeddings.shape)
    print("embedding ends, starting eval")
    score = eval(model, embeddings, test_loader,all_corpus)
    print("the eval score is "+str(score))






