import re
import torch
from transformers import BertTokenizer
from config import Args


class InputExample:
    """
    保存原始输入数据的类
    """

    def __init__(self, set_type, text, seq_label, token_label):
        self.set_type = set_type  # 数据集类型(train/dev/test)
        self.text = text  # 输入文本
        self.seq_label = seq_label  # 序列标签(意图)
        self.token_label = token_label  # 词符标签(槽位)


class InputFeature:
    """
    保存处理后的特征的类
    """

    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 seq_label_ids,
                 token_label_ids):
        self.input_ids = input_ids  # 输入ID序列
        self.attention_mask = attention_mask  # 注意力掩码
        self.token_type_ids = token_type_ids  # 词符类型ID
        self.seq_label_ids = seq_label_ids  # 序列标签ID
        self.token_label_ids = token_label_ids  # 词符标签ID


class Processor:
    """
    数据处理器
    """

    @classmethod
    def get_examples(cls, path, set_type):
        """
        从文件读取原始数据并转换为InputExample对象
        """
        raw_examples = []
        with open(path, 'r') as fp:
            data = eval(fp.read())
        for i, d in enumerate(data):
            text = d['text']
            seq_label = d['intent']
            token_label = d['slots']
            raw_examples.append(
                InputExample(
                    set_type,
                    text,
                    seq_label,
                    token_label
                )
            )
        return raw_examples


def convert_example_to_feature(ex_idx, example, tokenizer, config):
    """
    将单个InputExample转换为InputFeature
    """
    set_type = example.set_type
    text = example.text
    seq_label = example.seq_label
    token_label = example.token_label

    # 将序列标签转换为ID
    seq_label_ids = config.seqlabel2id[seq_label]

    # 初始化词符标签ID
    token_label_ids = [0] * len(text)
    for k, v in token_label.items():
        re_res = re.finditer(v, text)
        for span in re_res:
            entity = span.group()
            start = span.start()
            end = span.end()
            token_label_ids[start] = config.nerlabel2id['B-' + k]
            for i in range(start + 1, end):
                token_label_ids[i] = config.nerlabel2id['I-' + k]

    # 处理token_label_ids的长度
    if len(token_label_ids) >= config.max_len - 2:
        token_label_ids = [0] + token_label_ids[:config.max_len - 2] + [0]
    else:
        token_label_ids = [0] + token_label_ids + [0] + [0] * (config.max_len - len(token_label_ids) - 2)

    # 使用tokenizer处理文本
    text = [i for i in text]
    inputs = tokenizer.encode_plus(
        text=text,
        max_length=config.max_len,
        padding='max_length',
        truncation='only_first',
        return_attention_mask=True,
        return_token_type_ids=True,
    )

    # 转换为张量
    input_ids = torch.tensor(inputs['input_ids'], requires_grad=False)
    attention_mask = torch.tensor(inputs['attention_mask'], requires_grad=False)
    token_type_ids = torch.tensor(inputs['token_type_ids'], requires_grad=False)
    seq_label_ids = torch.tensor(seq_label_ids, requires_grad=False)
    token_label_ids = torch.tensor(token_label_ids, requires_grad=False)

    # 打印前三个样本的处理结果
    if ex_idx < 3:
        print(f'*** {set_type}_example-{ex_idx} ***')
        print(f'text: {text}')
        print(f'input_ids: {input_ids}')
        print(f'attention_mask: {attention_mask}')
        print(f'token_type_ids: {token_type_ids}')
        print(f'seq_label_ids: {seq_label_ids}')
        print(f'token_label_ids: {token_label_ids}')

    feature = InputFeature(
        input_ids,
        attention_mask,
        token_type_ids,
        seq_label_ids,
        token_label_ids,
    )

    return feature


def get_features(raw_examples, tokenizer, args):
    """
    将所有InputExample转换为InputFeature
    """
    features = []
    for i, example in enumerate(raw_examples):
        feature = convert_example_to_feature(i, example, tokenizer, args)
        features.append(feature)
    return features


if __name__ == '__main__':
    args = Args()
    raw_examples = Processor.get_examples('./data/test_process.json', 'test')
    tokenizer = BertTokenizer.from_pretrained('../../model_hub/chinese-bert-wwm-ext/')
    features = get_features(raw_examples, tokenizer, args)