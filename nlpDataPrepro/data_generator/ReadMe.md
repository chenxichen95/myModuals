## 目标：
  基于 tf.data.Dataset.from_generator ，针对阅读理解任务或问答任务，写一个通用的数据生成类。

## 主要模块：
  + 文本预处理
  + 文本分词
  + 构建单词表
  + 文本 index 化
  + 可以根据当前 batch 中句子的最大词数，对其他长度小于最大词数的句子进行 padding
  + 提供指定 batch size 的数据生成 api

## 注意：
  该模板不区分训练集、验证集和测试集。方便后续的拓展。

## Demo data
  使用一个常见的，简单的数据 Demo 来进行类的构建，方便后续特定任务的拓展。
  ~~~
      data = [
        {'id': 1, 'query': 'what is it ?', 'answer': 'this is a cat .'},
        {'id': 2, 'query': 'how are you?', 'answer': 'fine .'},
        {'id': 3, 'query': 'how can i do it ?', 'answer': 'you can pick it up .'},
      ]
  ~~~
