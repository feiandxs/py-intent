# 导入必要的库
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
import os
from seqeval.metrics.sequence_labeling import get_entities

# 导入自定义模块
from config import Args
from model import BertForIntentClassificationAndSlotFilling
from dataset import BertDataset
from preprocess import Processor, get_features


class Trainer:
    def __init__(self, model, config):
        # 初始化训练器
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()  # 定义损失函数
        self.optimizer = Adam(self.model.parameters(), lr=config.lr)  # 定义优化器
        self.epoch = args.epoch  # 训练轮数
        self.device = args.device  # 使用的设备（CPU或GPU）

    def train(self, train_loader):
        # 训练模型
        global_step = 0
        total_step = len(train_loader) * self.epoch
        self.model.train()
        for epoch in range(self.epoch):
            for step, train_batch in enumerate(train_loader):
                # 将数据移到指定设备
                for key in train_batch.keys():
                    train_batch[key] = train_batch[key].to(self.device)

                # 解析输入数据
                input_ids = train_batch['input_ids']
                attention_mask = train_batch['attention_mask']
                token_type_ids = train_batch['token_type_ids']
                seq_label_ids = train_batch['seq_label_ids']
                token_label_ids = train_batch['token_label_ids']

                # 前向传播
                seq_output, token_output = self.model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                )

                # 计算损失
                active_loss = attention_mask.view(-1) == 1
                active_logits = token_output.view(-1, token_output.shape[2])[active_loss]
                active_labels = token_label_ids.view(-1)[active_loss]

                seq_loss = self.criterion(seq_output, seq_label_ids)
                token_loss = self.criterion(active_logits, active_labels)
                loss = seq_loss + token_loss

                # 反向传播和优化
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f'[train] epoch:{epoch + 1} {global_step}/{total_step} loss:{loss.item()}')
                global_step += 1

        # 保存模型
        if self.config.do_save:
            self.save(self.config.save_dir, 'model.pt')

    def test(self, test_loader):
        # 测试模型
        self.model.eval()
        seq_preds = []
        seq_trues = []
        token_preds = []
        token_trues = []
        with torch.no_grad():
            for step, test_batch in enumerate(test_loader):
                # 将数据移到指定设备
                for key in test_batch.keys():
                    test_batch[key] = test_batch[key].to(self.device)

                # 解析输入数据
                input_ids = test_batch['input_ids']
                attention_mask = test_batch['attention_mask']
                token_type_ids = test_batch['token_type_ids']
                seq_label_ids = test_batch['seq_label_ids']
                token_label_ids = test_batch['token_label_ids']

                # 前向传播
                seq_output, token_output = self.model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                )

                # 处理序列输出
                seq_output = seq_output.detach().cpu().numpy()
                seq_output = np.argmax(seq_output, -1)
                seq_label_ids = seq_label_ids.detach().cpu().numpy()
                seq_label_ids = seq_label_ids.reshape(-1)
                seq_preds.extend(seq_output)
                seq_trues.extend(seq_label_ids)

                # 处理token输出
                token_output = token_output.detach().cpu().numpy()
                token_label_ids = token_label_ids.detach().cpu().numpy()
                token_output = np.argmax(token_output, -1)
                active_len = torch.sum(attention_mask, -1).view(-1)
                for length, t_output, t_label in zip(active_len, token_output, token_label_ids):
                    t_output = t_output[1:length - 1]
                    t_label = t_label[1:length - 1]
                    t_ouput = [self.config.id2nerlabel[i] for i in t_output]
                    t_label = [self.config.id2nerlabel[i] for i in t_label]
                    token_preds.append(t_ouput)
                    token_trues.append(t_label)

        # 计算并打印评估指标
        acc, precision, recall, f1 = self.get_metrices(seq_trues, seq_preds, 'cls')
        report = self.get_report(seq_trues, seq_preds, 'cls')
        ner_acc, ner_precision, ner_recall, ner_f1 = self.get_metrices(token_trues, token_preds, 'ner')
        ner_report = self.get_report(token_trues, token_preds, 'ner')
        print('意图识别：\naccuracy:{}\nprecision:{}\nrecall:{}\nf1:{}'.format(
            acc, precision, recall, f1
        ))
        print(report)
        print('槽位填充：\naccuracy:{}\nprecision:{}\nrecall:{}\nf1:{}'.format(
            ner_acc, ner_precision, ner_recall, ner_f1
        ))
        print(ner_report)

    def get_metrices(self, trues, preds, mode):
        # 计算评估指标
        if mode == 'cls':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(trues, preds)
            precision = precision_score(trues, preds, average='micro')
            recall = recall_score(trues, preds, average='micro')
            f1 = f1_score(trues, preds, average='micro')
        elif mode == 'ner':
            from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(trues, preds)
            precision = precision_score(trues, preds)
            recall = recall_score(trues, preds)
            f1 = f1_score(trues, preds)
        return acc, precision, recall, f1

    def get_report(self, trues, preds, mode):
        # 生成分类报告
        if mode == 'cls':
            from sklearn.metrics import classification_report
            report = classification_report(trues, preds)
        elif mode == 'ner':
            from seqeval.metrics import classification_report
            report = classification_report(trues, preds)
        return report

    def save(self, save_path, save_name):
        # 保存模型
        torch.save(self.model.state_dict(), os.path.join(save_path, save_name))

    def predict(self, text):
        # 对单个文本进行预测
        self.model.eval()
        with torch.no_grad():
            tmp_text = [i for i in text]
            inputs = tokenizer.encode_plus(
                text=tmp_text,
                max_length=self.config.max_len,
                padding='max_length',
                truncation='only_first',
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            token_type_ids = inputs['token_type_ids']
            seq_output, token_output = self.model(
                input_ids,
                attention_mask,
                token_type_ids,
            )
            seq_output = seq_output.detach().cpu().numpy()
            token_output = token_output.detach().cpu().numpy()
            seq_output = np.argmax(seq_output, -1)
            token_output = np.argmax(token_output, -1)
            seq_output = seq_output[0]
            token_output = token_output[0][1:len(text) + 1]
            token_output = [self.config.id2nerlabel[i] for i in token_output]
            print('意图：', self.config.id2seqlabel[seq_output])
            print('槽位：', str([(i[0], text[i[1]:i[2] + 1], i[1], i[2]) for i in get_entities(token_output)]))


if __name__ == '__main__':
    # 主函数，用于设置参数和执行训练、测试、预测
    args = Args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    if args.do_train:
        # 准备训练数据
        raw_examples = Processor.get_examples(args.train_path, 'train')
        train_features = get_features(raw_examples, tokenizer, args)
        train_dataset = BertDataset(train_features)
        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)

    if args.do_eval:
        # 准备评估数据
        raw_examples = Processor.get_examples(args.test_path, 'test')
        test_features = get_features(raw_examples, tokenizer, args)
        test_dataset = BertDataset(test_features)
        test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True)

    if args.do_test:
        # 准备测试数据
        raw_examples = Processor.get_examples(args.test_path, 'test')
        test_features = get_features(raw_examples, tokenizer, args)
        test_dataset = BertDataset(test_features)
        test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True)

    # 初始化模型
    model = BertForIntentClassificationAndSlotFilling(args)

    if args.load_model:
        # 加载预训练模型
        model.load_state_dict(torch.load(args.load_dir))

    model.to(device)
    trainer = Trainer(model, args)

    if args.do_train:
        # 训练模型
        trainer.train(train_loader)

    if args.do_test:
        # 测试模型
        trainer.test(test_loader)

    if args.do_predict:
        # 使用模型进行预测
        with open('./data/test.json', 'r') as fp:
            pred_data = eval(fp.read())
            for i, p_data in enumerate(pred_data):
                text = p_data['text']
                print('=================================')
                print(text)
                trainer.predict(text)
                print('=================================')
                if i == 10:
                    break