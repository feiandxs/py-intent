
## 数据集说明

项目使用了以下数据文件：

1. `domains.txt`: 包含所有支持的领域列表
2. `intents.txt`: 包含所有可能的意图列表
3. `slots.txt`: 包含所有可能的槽位类型
4. `sentences.txt`: 包含一些示例句子
5. `train.json`: 训练数据集
6. `train_process.json`: 处理后的训练数据集
7. `test.json`: 测试数据集
8. `test_process.json`: 处理后的测试数据集

## 系统架构

系统使用了基于BERT的神经网络模型，同时进行意图分类和槽位填充任务。模型架构定义在`BertForIntentClassificationAndSlotFilling`类中。

模型如果联网下载不了需要提前下载。


数据预处理
数据格式转换：
将原始的JSON格式数据转换为模型可以处理的格式。
使用domains.txt、intents.txt和slots.txt创建标签到索引的映射。
分词和编码：
使用BERT tokenizer对文本进行分词。
将词符、意图和槽位标签转换为对应的索引。
创建数据加载器：
使用PyTorch的DataLoader创建训练和测试数据加载器。
训练步骤
模型初始化：
使用BertForIntentClassificationAndSlotFilling类初始化模型。
设置训练参数：
定义批次大小、学习率、训练轮数等超参数。
训练循环：
遍历训练数据集。
对每个批次进行前向传播、损失计算和反向传播。
更新模型参数。
验证：
在每个epoch结束后，在验证集上评估模型性能。
保存模型：
保存训练好的模型权重。
评估和测试
加载测试数据：
使用test.json或test_process.json加载测试数据。
模型推理：
对测试数据进行预测，得到意图和槽位。
性能评估：
计算意图分类的准确率。
计算槽位填充的F1分数。