"""
Jan 2020
Xinru Yan

"""
import numpy as np
import config
from typing import List, Optional
import csv


class Examples:
    def __init__(self, conversations, labels, lengths):
        self.conversations = conversations
        self.labels = labels
        self.lengths = lengths

        assert len(self.conversations) == len(self.labels) == len(self.lengths), "There must be the same number of conversations as labels and lengths"

    def __add__(self, other: 'Examples'):
        assert isinstance(other, Examples), f'You can only add two example together not {type(other)} and Example'
        return Examples(self.conversations+other.conversations, self.labels+other.labels, self.lengths+other.lengths)

    def __len__(self):
        return len(self.conversations)

    def __iter__(self):
        return iter([self.conversations, self.labels, self.lengths])

    def shuffled(self):
        c = list(zip(self.conversations, self.labels, self.lengths))
        np.random.shuffle(c)
        return Examples(*zip(*c))

    def get_conversations(self, indexes: Optional[List[int]] = None):
        if indexes is None:
            return self.conversations
        else:
            return [x for i, x in enumerate(self.conversations) if i in indexes]

    def get_labels(self, indexes: Optional[List[int]] = None):
        if indexes is None:
            return self.labels
        else:
            return [x for i, x in enumerate(self.labels) if i in indexes]


class DataLoader:
    def __init__(self, config):
        self.config = config

        self.UNK = '<UNK>'
        self.PAD = '<PAD>'

        self.train_file_path = config.train_file_path
        self.dev_file_path = config.dev_file_path
        self.test_file_path = config.test_file_path

        print('building vocabularies from all train, dev and test')
        self.w2i, self.i2w, self.l2i, self.i2l = self.build_vocabs(self.train_file_path, self.dev_file_path,
                                                                   self.test_file_path)
        self.UNK_IDX = self.w2i[self.UNK]
        self.PAD_IDX = self.w2i[self.PAD]

        self.vocab_size = len(self.w2i)
        self.label_size = len(self.l2i)
        print(f'vocab size is {self.vocab_size}, label size is {self.label_size}')

        print('train and eval on train + dev')
        train_examples = self.load_data(self.train_file_path)
        dev_examples = self.load_data(self.test_file_path)

        print('padding')
        self.dev_examples = self.pad_conversations(dev_examples)
        self.train_examples = self.pad_conversations(train_examples)
        print('test on test')
        self.test_examples = self.pad_conversations(self.load_data(self.dev_file_path))

    def build_vocabs(self, *file_paths):
        words_set = set()
        labels = set()

        for file in file_paths:
            with open(file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    label = row['donation_label']
                    words = row['text']
                    labels.add(label)
                    for word in words.strip().split():
                        words_set.add(word)

        word_list = list(words_set)
        w2i = {word: i for i, word in enumerate(word_list)}
        w2i[self.UNK] = len(w2i)
        w2i[self.PAD] = len(w2i)
        i2w = {i: word for word, i in w2i.items()}

        l2i = {label: i for i, label in enumerate(list(labels))}
        i2l = {i: label for label, i in l2i.items()}

        return w2i, i2w, l2i, i2l

    def load_data(self, *file_paths) -> Examples:
        labels = []
        conversations = []
        lengths = []

        for file in file_paths:
            with open (file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    label = row['donation_label']
                    words = row['text']
                    labels.append(self.l2i[label.strip()])
                    sent = []
                    for word in words.strip().split():
                        sent.append(self.w2i[word] if word in self.w2i else self.UNK_IDX)
                    conversations.append(sent)
                    lengths.append(len(sent))
            assert len(conversations) == len(labels) == len(lengths)
        return Examples(conversations=conversations, labels=labels, lengths=lengths)

    def pad_conversations(self, examples: Examples) -> Examples:
        labels = examples.labels
        conversations = examples.conversations
        lengths = examples.lengths
        max_len = max(len(sent) for sent in conversations)
        padded_conversations = [sent if len(sent) == max_len else sent + [self.PAD_IDX] * (max_len - len(sent)) for sent in conversations]
        assert [len(sent) == max_len for sent in conversations]
        return Examples(conversations=padded_conversations, labels=labels, lengths=lengths)


def batch(examples: Examples, batch_size: int):
    batches = []
    batch_num = int(np.ceil(len(examples) / float(batch_size)))
    for i in range(batch_num):
        batch_conversations = examples.conversations[i*batch_size:(i+1)*batch_size]
        batch_labels = examples.labels[i*batch_size:(i+1)*batch_size]
        batch_lengths = examples.lengths[i*batch_size:(i+1)*batch_size]
        batch_data = Examples(batch_conversations, batch_labels, batch_lengths)
        batches.append(batch_data)

    for b in batches:
        yield b


def load_pte(pte, frequent):
    '''
    load pte from a text file.
    :param pte: path to pte
    :param frequent: most frequent n embedding
    :return: embedding, w2i
    '''
    vectors = []
    w2i = {}

    emb_dim = 300
    with open(pte, 'r', encoding='utf-8') as f:
        for i, row in enumerate(f):
            if i == 0:
                # no need to read first line
                pass
            else:
                word, vec = row.rstrip().split(' ', 1)
                word = word.lower()
                vec = np.fromstring(vec, sep=' ')
                if word not in w2i:
                    assert vec.shape == (emb_dim,), i
                    w2i[word] = len(w2i)
                    vectors.append(vec[None])
                else:
                    pass
            # only load the most frequent
            if len(w2i) >= frequent:
                break

    assert len(w2i) == len(vectors)

    embeddings = np.concatenate(vectors, 0)
    return embeddings, w2i


# def main():
#     dl = DataLoader(config)
#     b = batch(dl.train_examples, config.batch_size)
#
#     for bb in b:
#         print(bb.conversations)
#         print(bb.labels)
#         print(bb.lengths)
#         #print(list(bb))
#         break
#
# if __name__ == '__main__':
#     main()
