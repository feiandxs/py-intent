import torch.nn as nn
from transformers import BertModel
from transformers import BertForTokenClassification


class BertForIntentClassificationAndSlotFilling(nn.Module):
    """
    用于意图分类和槽位填充的BERT模型
    """

    def __init__(self, config):
        """
        初始化模型

        参数:
        config: 包含模型配置的对象
        """
        super(BertForIntentClassificationAndSlotFilling, self).__init__()
        self.config = config

        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(config.bert_dir)
        self.bert_config = self.bert.config

        # 用于意图分类（序列分类）的层
        self.sequence_classification = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.seq_num_labels),
        )

        # 用于槽位填充（词符分类）的层
        self.token_classification = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.token_num_labels),
        )

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids):
        """
        前向传播

        参数:
        input_ids: 输入的词ID序列
        attention_mask: 注意力掩码
        token_type_ids: 词符类型ID

        返回:
        seq_output: 意图分类的输出
        token_output: 槽位填充的输出
        """
        # 获取BERT的输出
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        pooler_output = bert_output[1]  # [CLS]词符的输出，用于序列分类
        token_output = bert_output[0]  # 所有词符的输出，用于词符分类

        # 意图分类
        seq_output = self.sequence_classification(pooler_output)

        # 槽位填充
        token_output = self.token_classification(token_output)

        return seq_output, token_output