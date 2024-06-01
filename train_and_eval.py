import re

import torch
from sentence_transformers import SentenceTransformer,util
from torch import optim, autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from Datas.QA_dataset import QADataset
from Datas.big_dataset import BIGDataset
from utils.baseloss import MultiDSCLoss
from utils.model_utils import XFBert
from sklearn.metrics import f1_score


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



def eval(pretrain_model, model,test_loader):
    precision = 0

    pred = []
    gt = []

    model.load_state_dict(torch.load("bert.pth"))

    model.eval()
    eval_loop = tqdm(test_loader)
    for query in test_loader:
        # questions.append(query[0])
        # answers.append(query[1])
        text = query[0]+query[1] #query[0]为问题， query[1]为依据, query[3]为"Yes"或"No"
        label = query[2]
        # text = model.encode(text, convert_to_tensor=True).to('cuda') 需要embedding加下面这句

        output = model(text)

        if label == 'Yes':
            if output == 1:
                pred.append(1)
                precision+=1
            else:
                pred.append(0)
            gt.append(1)
        else:
            if output == 1:
                pred.append(1)
            else:
                pred.append(0)
                precision += 1
            gt.append(0)


        f1 = f1_score(gt,pred)

        # query_embedding = model.encode(question, convert_to_tensor=True).to('cuda')
        #
        # similarity_scores = model.similarity(query_embedding, corpus_embeddings)[0]
        # scores, indices = torch.topk(similarity_scores, k=1)
        # result = indices[0]
        #
        #
        # f1 = getF1_score(result, answer)
        # score += f1/len(test_loader)
    # query_embeddings = model.encode(questions)
    # hits = util.semantic_search(query_embeddings, corpus_embeddings)
    # for i in range(hits):
    #     result = corpus(hits[i]['corpus_id'])
    #     f1 = getF1_score(result, answers[i])
    #     score += f1/len(test_loader)



    return score/len(test_loader),f1_score

def train(dataloader,pre_train_model, model, epoch, optimizer, loss):
    iterator = tqdm(range(dataloader), ncols=70)
    device = 'cuda'
    model.to('cuda')
    intent_loss_fct = MultiDSCLoss(alpha=0.3, smooth=1.0)
    scaler = GradScaler()

    for i in range(epoch):
        model.train()
        epochs_loss = 0
        optimizer.zero_grad()
        for step, batch in tqdm(enumerate(dataloader, start=1)):
            with autocast():
                # text[0]为问题， text[1]为依据, text[3]为正确或错误

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                intent_labels = batch['intent_labels'].to(device)
                intent_logits = model()
                intent_loss = intent_loss_fct(intent_logits.view(-1, 2), intent_labels)
                loss = intent_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()

    print('save model ...')
    torch.save(model.state_dict(), "bert.pth")



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
    precision, f1_score = eval(pre_train_model, model, test_loader)
    print("the eval precision is "+str(precision)+" f1 score is "+str(f1_score))






