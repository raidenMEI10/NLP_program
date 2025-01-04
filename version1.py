
import re
import torch
import torch.nn as nn  # 导入 nn 模块
from torch import optim, autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaForSequenceClassification, \
    RobertaModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset
from utils.baseloss import MultiDSCLoss


# 定义处理字符串的函数
def process_string(s, n_number):
    s = re.sub(r'\d*', '', s, count=len(str(n_number)))
    s = re.sub(r'\s*', '', s, count=1)
    result_list = re.split(r'(?<!\d)\. (?!\d)', s)
    return result_list


# 定义从文件连续读取的函数
def read_file_continuously(file_path, index):
    alltext = []
    try:
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


# 定义 QADataset 类
class QADataset(Dataset):
    def __init__(self, file_path, num_samples):
        self.questions = []
        self.evidences = []
        self.labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:num_samples]:
                parts = line.strip().split('\t')
                if len(parts) == 5:
                    self.questions.append(parts[1])
                    self.evidences.append(parts[3])
                    self.labels.append(1 if parts[2] == 'Yes' else 0)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.evidences[idx], self.labels[idx]


# 定义 collate_fn 函数
def collate_fn(batch):
    questions, evidences, labels = zip(*batch)
    return list(questions), list(evidences), torch.tensor(labels)


# 定义 create_inputs 函数
def create_inputs(question, evidence, tokenizer, device, max_length=512):
    text = question + " [SEP] " + evidence
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(
        device)
    return inputs


# 定义 CustomRobertaForSequenceClassification 类
class CustomRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.classifier = nn.Linear(config.hidden_size + 5000, config.num_labels)  # 5000 是 TF-IDF 特征的数量

    def forward(self, input_ids=None, attention_mask=None, tfidf_features=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0][:, 0, :]  # 获取 [CLS] token 的输出

        # 将 BERT 输出与 TF-IDF 特征连接起来
        combined_features = torch.cat((sequence_output, tfidf_features), dim=1)

        logits = self.classifier(combined_features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


# 定义 eval 函数
def eval(pretrain_model, model, tokenizer, device, data_loader, tfidf_vectorizer):
    model.eval()
    total_loss = 0
    pred = []
    gt = []

    loss_fn = MultiDSCLoss(alpha=0.3, smooth=1.0)

    with torch.no_grad():
        for batch in tqdm(data_loader):
            questions, evidences, labels = batch
            inputs_list = [create_inputs(q, e, tokenizer, device) for q, e in zip(questions, evidences)]
            inputs = {key: torch.cat([inp[key] for inp in inputs_list], dim=0) for key in inputs_list[0]}
            tfidf_features = tfidf_vectorizer.transform([q + " " + e for q, e in zip(questions, evidences)]).toarray()
            tfidf_features = torch.tensor(tfidf_features, dtype=torch.float32).to(device)
            labels = labels.to(device)
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                            tfidf_features=tfidf_features)
            loss = loss_fn(outputs[0].view(-1, 2), labels)
            total_loss += loss.item()
            predictions = torch.argmax(outputs[0], dim=1).cpu().numpy()
            pred.extend(predictions)
            gt.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(gt, pred)
    f1 = f1_score(gt, pred)
    print("Evaluation finished...")
    return avg_loss, accuracy, f1


# 定义 train 函数
def train(dataloader, pre_train_model, model, tokenizer, epoch, optimizer, loss_fn, device, test_loader,
          tfidf_vectorizer):
    train_losses = []
    test_losses = []
    test_accuracies = []
    test_f1_scores = []
    intent_loss_fct = MultiDSCLoss(alpha=0.3, smooth=1.0)
    scaler = GradScaler()
    best_accuracy = 0

    for ep in range(epoch):
        model.train()
        total_train_loss = 0
        for step, batch in tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc=f'Epoch {ep + 1}/{epoch}'):
            optimizer.zero_grad()
            questions, evidences, labels = batch
            inputs_list = [create_inputs(q, e, tokenizer, device) for q, e in zip(questions, evidences)]
            inputs = {key: torch.cat([inp[key] for inp in inputs_list], dim=0) for key in inputs_list[0]}
            tfidf_features = tfidf_vectorizer.transform([q + " " + e for q, e in zip(questions, evidences)]).toarray()
            tfidf_features = torch.tensor(tfidf_features, dtype=torch.float32).to(device)
            labels = labels.to(device)
            with autocast(device_type='cuda'):
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                tfidf_features=tfidf_features)
                loss = intent_loss_fct(outputs[0].view(-1, 2), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(dataloader)
        train_losses.append(avg_train_loss)

        avg_test_loss, test_accuracy, test_f1 = eval(pre_train_model, model, tokenizer, device, test_loader,
                                                     tfidf_vectorizer)

        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)
        test_f1_scores.append(test_f1)

        # 保存最优模型权重
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {ep + 1}/{epoch}")
        print(f"Training Loss: {avg_train_loss}")
        print(f"Test Loss: {avg_test_loss}")
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Test F1 Score: {test_f1}")

    return train_losses, test_losses, test_accuracies, test_f1_scores, best_accuracy


# 重新运行主要代码块
if __name__ == '__main__':
    model = CustomRobertaForSequenceClassification.from_pretrained("C:/Users/20163/Desktop/大三下文件夹/实践——NLP/embedding_method/roberta3_0",
                                                                   num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("C:/Users/20163/Desktop/大三下文件夹/实践——NLP/embedding_method/roberta3_0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    pre_train_model = SentenceTransformer("./all-MiniLM-L6-v2", device='cuda')

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    train_dataset = QADataset('./Datas/train.tsv', 10053)
    test_dataset = QADataset('./Datas/test.tsv', 3080)

    # 初始化 TF-IDF 向量器
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    # 创建训练集和测试集的文本数据列表
    train_texts = [q + " " + e for q, e in zip(train_dataset.questions, train_dataset.evidences)]
    test_texts = [q + " " + e for q, e in zip(test_dataset.questions, test_dataset.evidences)]

    # 使用 TF-IDF 向量器拟合训练集数据
    tfidf_vectorizer.fit(train_texts + test_texts)

    # 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)

    intent_loss_fct = MultiDSCLoss(alpha=0.3, smooth=1.0)

    print("beginning to read files...")
    print("read is over, next is training...")
    train_losses, test_losses, test_accuracies, test_f1_scores, best_accuracy = train(train_loader, pre_train_model,
                                                                                      model, tokenizer,
                                                                                      epoch=8, optimizer=optimizer,
                                                                                     loss_fn=intent_loss_fct,device=device,
                                                                                      test_loader=test_loader,
                                                                                      tfidf_vectorizer=tfidf_vectorizer)

    print("training ends, loading best model for final evaluation")
    model.load_state_dict(torch.load("best_model.pth"))
    avg_test_loss, test_accuracy, test_f1 = eval(pre_train_model, model, tokenizer, device, test_loader,
                                                 tfidf_vectorizer)
    print(f"Final Evaluation - Best Test Accuracy: {best_accuracy}")
    print(f"Final Evaluation - Test Loss: {avg_test_loss}, Test Accuracy: {test_accuracy}, Test F1 Score: {test_f1}")


#python /root/autodl-fs/embedding_method/version2.py