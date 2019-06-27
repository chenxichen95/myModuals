import tensorflow as tf
import os

class DataLoader(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.raw_data = self.load_raw_data()
        self.vocab_w2i, self.vocab_i2w = self.build_vocab()

        self.dataset = ''  # the object of tf.data.Dataset.from_generator
        self.iterator = ''
        self.init_op = ''  # the op used for initiating the generator
        self.next_batch_op = ''  # the op used for getting the next batch data

    def load_raw_data(self):
        '''
            load raw data
        '''
        raw_data = [
            {'id': 1, 'query': 'what is it ?', 'answer': 'this is a cat .'},
            {'id': 2, 'query': 'how are you?', 'answer': 'fine .'},
            {'id': 3, 'query': 'how can i do it ?', 'answer': 'you can pick it up .'},
        ]
        return raw_data

    def text_preprocess(self, text):
        '''
            preprocess the text, like filter, drop stop word ..
        '''
        return text

    def build_vocab(self):
        '''
            build vocab from the text in the raw data
        '''
        vocab_w2i = {}
        vocab_i2w = {}
        vocab_w2i['<pad>'] = 0
        vocab_w2i['<unk>'] = 1
        vocab_i2w[0] = '<pad>'
        vocab_i2w[1] = '<unk>'
        for item in self.raw_data:
            for word in self.split_word(item['query']):
                if vocab_w2i.get(word, None) is None:
                    idx = len(vocab_w2i)
                    vocab_w2i[word] = idx
                    vocab_i2w[idx] = word
            for word in self.split_word(item['answer']):
                if vocab_w2i.get(word, None) is None:
                    idx = len(vocab_w2i)
                    vocab_w2i[word] = idx
                    vocab_i2w[idx] = word

        return vocab_w2i, vocab_i2w

    def split_word(self, sent):
        '''
            split word
        '''
        return sent.split(' ')

    def encode(self, sent):
        sent = sent.decode('utf-8')
        sent = self.text_preprocess(sent)
        sent = self.split_word(sent)
        return [self.vocab_w2i.get(word, self.vocab_w2i['<unk>']) for word in sent]

    def gen(self, query, answer):
        '''
            the function of generator used for tf.data.Dataset.from_generator
        '''
        for q, a in zip(query, answer):
            q_en = self.encode(q)
            a_en = self.encode(a)
            yield (q_en, a_en)

    def create_generator(self):
        query = [item['query'] for item in self.raw_data]
        answer = [item['answer'] for item in self.raw_data]

        # create tf.data.Dataset
        types = (tf.int32, tf.int32)
        shapes = ([None], [None])
        dataset = tf.data.Dataset.from_generator(
            generator=self.gen,
            output_types=types,
            output_shapes=shapes,
            args=(query, answer)
        )
        dataset = dataset.repeat()

        # setting the paddings for each batch data
        # 会根据当前 batch 的最大长度进行 padding
        paddings = (0, 0)
        dataset = dataset.padded_batch(self.batch_size, shapes, paddings).prefetch(1)

        # create tf.data.Iterator
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        self.next_batch_op = iterator.get_next()
        self.init_op = iterator.make_initializer(dataset)
        self.dataset = dataset
        self.iterator = iterator


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True)
    session_conf.gpu_options.allow_growth = True

    with tf.Session(config=session_conf) as sess:

        # create dataloader
        dataloader = DataLoader(batch_size=2)
        dataloader.create_generator()

        # initiate the generator
        sess.run(dataloader.init_op)

        # get the next batch data
        data = []
        for i in range(4):
            data.append(sess.run(dataloader.next_batch_op))

    for i in range(4):
        print('=' * 20 + f'batch {i}' + '=' * 20)
        print('q')
        print(data[i][0])
        print('a')
        print(data[i][1])