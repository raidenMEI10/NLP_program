import re

import torch
from sentence_transformers import SentenceTransformer,util
from torch import optim, autocast
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from Datas.QA_dataset import QADataset
from Datas.big_dataset import BIGDataset
from utils.model_utils import XFBert


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

def eval(pretrain_model, model,test_loader):
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

def train(dataloader,pre_train_model, model, epoch, optimizer, loss):
    iterator = tqdm(range(dataloader), ncols=70)
    model.to('cuda')

    for i in range(epoch):
        model.train()
        epochs_loss = 0
        optimizer.zero_grad()
        for step, batch in tqdm(enumerate(dataloader, start=1)):
            with autocast():
                print(batch)
                intent_logits = model()
                # intent_loss = intent_loss_fct(intent_logits.view(-1, len(intent2id)), intent_labels)
                # loss = intent_loss

            # scaler.scale(loss).backward()



    # embeddings = pretrain_model.encode(, device='cuda').to("cuda")



if __name__ == '__main__':


    # 加载模型 (会添加针对特定任务类型的Head)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    # dict(model.named_children()).keys()
    #model = XFBert(r"./bert-base-chinese", rumor_dim=2, enable_mdrop=True)

    pre_train_model = SentenceTransformer("./all-MiniLM-L6-v2", device='cuda')

    optimizer = optim.SGD(model.parameters(), lr=1e-5)
    bce_loss = torch.nn.MSELoss()

    print("Max Sequence Length:", pre_train_model.max_seq_length)

    train_dataset = QADataset('./Datas/train.tsv', 10053)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=8)
    test_dataset = QADataset('./Datas/test.tsv',3080)
    # corpus_loader = torch.utils.data.DataLoader(corpus_dataset,batch_size = 32)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1)
    # corpus_loop = tqdm(corpus_loader, desc='Train')
    print("begining to read files...")
    #all_corpus = read_file_continuously('./Datas/doc.tsv', 2666763)
    # print("len of corpus is "+str(len(all_corpus)))
    print("read is over, next is training...")
    train(train_loader, pre_train_model, model, epoch=1000, optimizer=optimizer, loss=bce_loss)


    # print(embeddings.shape)
    print("training ends, starting eval")
    score = eval(pre_train_model, model, test_loader)
    print("the eval score is "+str(score))






