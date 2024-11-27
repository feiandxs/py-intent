from torch.utils.data import DataLoader, Dataset


class BertDataset(Dataset):
    """
    自定义的Dataset类,用于加载BERT模型的输入数据
    继承自torch.utils.data.Dataset
    """

    def __init__(self, features):
        """
        初始化Dataset

        参数:
        features: 预处理后的特征列表
        """
        self.features = features
        self.nums = len(self.features)

    def __len__(self):
        """
        返回数据集的大小
        """
        return self.nums

    def __getitem__(self, item):
        """
        返回一个数据样本

        参数:
        item: 索引

        返回:
        一个包含模型输入所需所有字段的字典
        """
        data = {
            'input_ids': self.features[item].input_ids.long(),
            'attention_mask': self.features[item].attention_mask.long(),
            'token_type_ids': self.features[item].token_type_ids.long(),
            'seq_label_ids': self.features[item].seq_label_ids.long(),
            'token_label_ids': self.features[item].token_label_ids.long(),
        }
        return data


if __name__ == '__main__':
    from config import Args
    from preprocess import Processor, get_features
    from transformers import BertTokenizer

    # 初始化配置
    args = Args()

    # 加载BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('../../model_hub/chinese-bert-wwm-ext/')

    # 处理训练数据
    raw_examples = Processor.get_examples('./data/train_process.json', 'train')
    train_features = get_features(raw_examples, tokenizer, args)
    train_dataset = BertDataset(train_features)
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)

    # 打印一个批次的训练数据,用于调试
    for step, train_batch in enumerate(train_loader):
        print(train_batch)
        break

    # 处理测试数据
    raw_examples = Processor.get_examples('./data/test_process.json', 'test')
    test_features = get_features(raw_examples, tokenizer, args)
    test_dataset = BertDataset(test_features)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True)