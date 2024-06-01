#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   trainer.py
@Contact :   littlepig@88.com

@Modify Time      @Author       @Version     @Desciption
------------      -------       --------     -----------
2023/5/7 21:34   littlepig        1.8           None
'''
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from torch.optim.swa_utils import AveragedModel,SWALR # SWA优化
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AdamW,get_scheduler
from tqdm import tqdm

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss
# from utils.ema_block import EMA
from utils.baseloss import PriorMultiLabelSoftMarginLoss,MultiFocalLoss,MultiDSCLoss

def get_optimizer_and_schedule(args,model,trainloader_shape):
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": args.learning_rate,
        },
    ]
    # initialize lrs for every layer
    num_layers = model.config.num_hidden_layers
    layers = [getattr(model, "model").embeddings] + list(getattr(model, "model").encoder.layer)
    layers.reverse()
    # print(layers)
    lr = args.learning_rate
    for layer in layers:
        lr *= 0.9
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

    # warmp_up步长
    num_training_steps = int(trainloader_shape / args.train_batch_size * args.epoches)
    num_warmup_steps = int(num_training_steps * args.warmup_proportion)
    # 优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay,eps=1e-6,correct_bias=True)
    if args.use_swa:
        scheduler = CosineAnnealingLR(optimizer, T_max=100)  # 使用学习率策略（余弦退火）
    else:
    # 学习策略
        scheduler = get_scheduler(args.scheduler_name,
                                  optimizer,
                                  num_warmup_steps= num_warmup_steps,
                                  num_training_steps=num_training_steps)

    return optimizer,scheduler
# 验证部分
def evaluation(args, model, data_loader, device, ema=None,labels_name=None):
    model.eval()
    if args.use_ema:
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
    labels = None
    prob_preds = None
    for batch in data_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent_labels = batch['intent_labels'].to(device)
            intent_logits = model(input_ids, attention_mask=attention_mask)
            intent_logits = F.softmax(intent_logits,dim=1)
            # 计算 slot 预测标签和真实标签
            if prob_preds is None:
                prob_preds = intent_logits.detach().cpu().numpy()
                labels = intent_labels.detach().cpu().numpy()
            else:
                prob_preds = np.vstack((prob_preds,intent_logits.detach().cpu().numpy()))
                labels = np.hstack((labels,intent_labels.detach().cpu().numpy()))
    intent_acc = accuracy_score(y_true=labels, y_pred=np.argmax(prob_preds,axis=1))
    preds = np.argmax(prob_preds, axis=1)
    precision = precision_score(y_true=labels, y_pred=preds, average='macro')
    recall = recall_score(y_true=labels, y_pred=preds, average='macro')
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    print("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(intent_acc, precision, recall, f1))
    # intent_auc = roc_auc_score(y_true=labels,y_score=prob_preds,multi_class='ovr',labels=labels_name)
    # print(intent_auc,intent_acc)
    if args.use_ema:
        ema.restore(model.parameters())
    return f1
def predict(args, model, data_loader, device, ema=None):
    model.eval()
    if args.use_ema:
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
    labels = None
    prob_preds = None
    for batch in tqdm(data_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent_labels = batch['intent_labels'].to(device)
            intent_logits = model(input_ids, attention_mask=attention_mask)
            intent_logits = F.softmax(intent_logits,dim=1)
            # 计算 slot 预测标签和真实标签
            if prob_preds is None:
                prob_preds = intent_logits.detach().cpu().numpy()
                labels = intent_labels.detach().cpu().numpy()
            else:
                prob_preds = np.vstack((prob_preds,intent_logits.detach().cpu().numpy()))
                labels = np.hstack((labels,intent_labels.detach().cpu().numpy()))
    intent_acc = accuracy_score(y_true=labels, y_pred=np.argmax(prob_preds,axis=1))
    preds = np.argmax(prob_preds, axis=1)
    precision = precision_score(y_true=labels, y_pred=preds, average='macro')
    recall = recall_score(y_true=labels, y_pred=preds, average='macro')
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    print("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(intent_acc, precision, recall, f1))
    if args.use_ema:
        ema.restore(model.parameters())
    return intent_acc

# 训练阶段
def do_train(args, model, train_dataloader, dev_dataloader, device, intent2id,optimizer, scheduler):
    total_step = len(train_dataloader) * args.epoches
    intent_model_total_epochs = 0
    best_score = 0.0
    this_epoch_training_loss = 0
    stop_nums = 0
    swa_start = 5
    iters_to_accumulate = args.grad_accumulate_nums
    # intent_loss_fct = CrossEntropyLoss()
    # intent_loss_fct = PriorMultiLabelSoftMarginLoss(prior=None, num_labels=len(intent2id))
    # intent_loss_fct = MultiFocalLoss(num_class=len(intent2id))
    intent_loss_fct = MultiDSCLoss(alpha=0.3, smooth=1.0)
    scaler = GradScaler()
    if args.use_ema:
        ema = EMA(model.parameters(),decay=args.ema_decay)
    else:
        ema = None
    if args.use_swa:
        swa_model = AveragedModel(model)
        # swa_start = 5  # 设置SWA开始的周期，当epoch到该值的时候才开始记录模型的权重
        swa_scheduler = SWALR(optimizer, swa_lr=1e-6)
    # 训练
    print("train ...")
    for epoch in range(0, args.epoches):
        if stop_nums >= args.early_stopping:
            break
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader, start=1)):
            with autocast():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                intent_labels = batch['intent_labels'].to(device)
                intent_logits = model(input_ids, attention_mask=attention_mask)
                intent_loss = intent_loss_fct(intent_logits.view(-1, len(intent2id)), intent_labels)
                loss = intent_loss

            scaler.scale(loss).backward()
            if (step + 1) % iters_to_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step()
                if args.use_ema:
                    ema.update(model.parameters())
                if args.use_swa:
                    if epoch > swa_start:
                        swa_model.update_parameters(model)
                        swa_scheduler.step()
                else:
                    scheduler.step()
                model.zero_grad()

            intent_model_total_epochs += 1

        if stop_nums >= args.early_stopping:
            break
        # 验证

        eval_intent_score = evaluation(args,model,dev_dataloader,device,ema=ema,labels_name=list(intent2id.keys()))
        # print("eval acc: %.5f" % eval_score)
        if best_score < eval_intent_score:
            stop_nums = 0
            print(r"【%.2f%%】 Intent F1-score update %.5f ---> %.5f " % ((intent_model_total_epochs/total_step),best_score, eval_intent_score))
            best_score = eval_intent_score
            if best_score > 0.5 :
                # 保存模型
                # model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model.state_dict(),args.save_dir_curr)
                # torch.save(model_to_save.state_dict(), os.path.join(args.save_dir_curr, "pytorch_model.bin"))
                # model_to_save.config.to_json_file(os.path.join(args.save_dir_curr, "config.json"))
        else:
            stop_nums = stop_nums + 1
            if stop_nums > int(args.early_stopping - 7):
                print("stop nums is {}".format(stop_nums))
    del ema
    return best_score
# 预测阶段
def do_sample_predict(model,data_loader,device,is_prob=False):
    model.eval()
    pred_intents = None
    for batch in data_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent_logits = model(input_ids, attention_mask=attention_mask)
            intent_logits = F.softmax(intent_logits, dim=1)
            if pred_intents is None:
                pred_intents = intent_logits.detach().cpu().numpy()
            else:
                pred_intents = np.vstack((pred_intents,intent_logits.detach().cpu().numpy()))
    if is_prob:
        return pred_intents,np.argmax(pred_intents,axis=1)
    return np.argmax(pred_intents,axis=1)
# 生成预测文件
def do_sample_sumbitfile(args,kfidx,predict_intent,id2intent):
    # 将意图标签id转换为文本
    pred_intent = [id2intent[i] for i in predict_intent]
    # 生成文件
    predict_submit = pd.DataFrame({"label":pred_intent})
    sumbit_excel_dir = os.path.join(args.results_excel_dir, args.MODEL_NAME)
    os.makedirs(sumbit_excel_dir, exist_ok=True)
    sumbit_excel_dir = os.path.join(sumbit_excel_dir,"sumbit_kflod{}.csv".format(kfidx))
    predict_submit.to_csv(sumbit_excel_dir,index=False)
    print(f"提交文件{sumbit_excel_dir}生成完毕")
# 生成预测文件
def do_sample_probfile(args,kfidx,predict_intent_prob):
    # 生成文件
    sumbit_excel_dir = os.path.join(args.results_excel_dir, args.MODEL_NAME)
    os.makedirs(sumbit_excel_dir, exist_ok=True)
    sumbit_excel_dir = os.path.join(sumbit_excel_dir,"prob_kflod{}.npy".format(kfidx))
    np.save(sumbit_excel_dir, predict_intent_prob)
    print(f"提交文件{sumbit_excel_dir}生成完毕")

def hard_voting(model_name_list,choose_file_list):
    intent_label = []
    voting_files = []
    search_dir = "../user_data/results"
    if len(model_name_list) > 1:
        voting_path = os.path.join(search_dir, "MultipleModel_HardVoting_Sumbit.csv")
    else:
        voting_path = os.path.join(search_dir,model_name_list[0],"sumbit_kflodhard.csv")
    for model_name,choose_file in zip(model_name_list,choose_file_list):
        choose_file = ["sumbit_kflod{}.csv".format(i) for i in choose_file]
        search_path = os.path.join(search_dir,model_name)
        voting_file = [os.path.join(search_path,file) for file in os.listdir(search_path) if file in choose_file]
        voting_files.extend(voting_file)
    print(voting_files)

    resluts = [pd.read_csv(file) for file in voting_files]
    for result in resluts:
        intent_label.append(result['label'].values.tolist())
    intent_label = pd.DataFrame(intent_label)
    hard_voting_intent = intent_label.mode().loc[0,:]
    voting_hard = pd.DataFrame({"label":hard_voting_intent})
    voting_hard.to_csv(voting_path,index=False)
    print(f"提交文件{voting_path}生成完毕")

def soft_voting(model_name_list,choose_file_list):
    id2intent = {0: '正向', 1: '负向'}
    voting_files = []
    search_dir = "../user_data/results"
    if len(model_name_list) > 1:
        voting_path = os.path.join(search_dir, "MultipleModel_SoftVoting_Sumbit.csv")
    else:
        voting_path = os.path.join(search_dir,model_name_list[0],"sumbit_kflodsoft.csv")
    for model_name,choose_file in zip(model_name_list,choose_file_list):
        choose_file = ["prob_kflod{}.npy".format(i) for i in choose_file]
        search_path = os.path.join(search_dir,model_name)
        voting_file = [os.path.join(search_path,file) for file in os.listdir(search_path) if file in choose_file]
        voting_files.extend(voting_file)
    print(voting_files)

    resluts = np.array([np.load(file) for file in voting_files])
    resluts = np.mean(resluts, axis=0)
    resluts = np.argmax(resluts,axis=1)
    intent_label = [id2intent[i] for i in resluts]
    voting_soft = pd.DataFrame({"label": intent_label})
    voting_soft.to_csv(voting_path,index=False)
    print(f"提交文件{voting_path}生成完毕")