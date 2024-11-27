class Args:
    # 数据文件路径
    train_path = './data/train_process.json'  # 训练数据文件路径
    test_path = './data/test_process.json'    # 测试数据文件路径
    seq_labels_path = './data/intents.txt'    # 序列标签(意图)文件路径
    token_labels_path = './data/slots.txt'    # 词符标签(槽位)文件路径

    # 模型相关路径
    bert_dir = '../../model_hub/chinese-bert-wwm-ext/'  # 预训练BERT模型目录
    save_dir = './checkpoints/'        # 模型保存目录
    load_dir = './checkpoints/model.pt'  # 模型加载文件路径

    # 运行模式控制
    do_train = False    # 是否进行训练
    do_eval = False     # 是否进行评估
    do_test = True      # 是否进行测试
    do_save = True      # 是否保存模型
    do_predict = True   # 是否进行预测
    load_model = True   # 是否加载已有模型

    # 设备设置
    device = None  # 计算设备,None表示自动选择(CPU/GPU)

    # 序列标签(意图)处理
    seqlabel2id = {}  # 序列标签到ID的映射
    id2seqlabel = {}  # ID到序列标签的映射
    with open(seq_labels_path, 'r') as fp:
        seq_labels = fp.read().split('\n')
        for i, label in enumerate(seq_labels):
            seqlabel2id[label] = i
            id2seqlabel[i] = label

    # 词符标签(槽位)处理
    tokenlabel2id = {}  # 词符标签到ID的映射
    id2tokenlabel = {}  # ID到词符标签的映射
    with open(token_labels_path, 'r') as fp:
        token_labels = fp.read().split('\n')
        for i, label in enumerate(token_labels):
            tokenlabel2id[label] = i
            id2tokenlabel[i] = label

    # NER标签处理(BIO标记方案)
    tmp = ['O']  # 'O'表示不属于任何实体
    for label in token_labels:
        B_label = 'B-' + label  # Begin标签
        I_label = 'I-' + label  # Inside标签
        tmp.append(B_label)
        tmp.append(I_label)
    nerlabel2id = {}  # NER标签到ID的映射
    id2nerlabel = {}  # ID到NER标签的映射
    for i, label in enumerate(tmp):
        nerlabel2id[label] = i
        id2nerlabel[i] = label

    # 模型参数
    hidden_size = 768  # 隐藏层大小,与BERT-base一致
    seq_num_labels = len(seq_labels)    # 序列标签(意图)的数量
    token_num_labels = len(tmp)         # NER标签的数量
    max_len = 32       # 输入序列的最大长度
    batchsize = 64     # 批次大小
    lr = 2e-5          # 学习率
    epoch = 10         # 训练轮数
    hidden_dropout_prob = 0.1  # 隐藏层的dropout概率

if __name__ == '__main__':
    args = Args()
    # 打印各种标签信息,用于调试和验证
    print("序列标签(意图):", args.seq_labels)
    print("序列标签到ID的映射:", args.seqlabel2id)
    print("词符标签到ID的映射:", args.tokenlabel2id)
    print("NER标签到ID的映射:", args.nerlabel2id)