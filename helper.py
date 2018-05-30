import tensorflow as tf
import numpy as np
import glob
import nltk
from collections import namedtuple

batch_size = 32


class Helper(namedtuple("Helper", ("initializer", "input_seq", "input_len", "target_seq", "target_len", "ix_to_char", "char_to_ix", "sos_id", "eos_id", "batch_size"))):
    pass


def build_helper():
    files = glob.glob('data/*.txt')
    # files = glob.glob('old_stuff/_FULL.txt')
    print('Loaded {} files'.format(len(files)))
    data_pairs = []
    vocab = set(['<', '>'])

    for f in files:
        data = open(f, 'r').read()
        vocab = vocab.union(data)
        sentences = nltk.sent_tokenize(data)
        # words = data.split()
        data_pairs += [(sentences[i]+'>', '<'+sentences[i+1]+'>') for i in range(len(sentences)-1)]
        # data_pairs += [(words[i], words[i+1]) for i in range(len(words)-1)]
    print('Created {} pairs'.format(len(data_pairs)))
    
    vocab = sorted(vocab)
    ix_to_char = { i: ch for i, ch in enumerate(vocab)}
    char_to_ix = { ch: i for i, ch in ix_to_char.items() }
    sos_id = char_to_ix['<']
    eos_id = char_to_ix['>']

    def generator():
        for p in data_pairs:
            d = ((np.array(list(map(char_to_ix.get, p[0]))), len(p[0])-1), (np.array(list(map(char_to_ix.get, p[1]))), len(p[1])-1))
            yield d

    ds = tf.data.Dataset.from_generator(generator, ((tf.int32, tf.int32), (tf.int32, tf.int32)))
    ds = ds.shuffle(1000)
    ds = ds.padded_batch(batch_size, padded_shapes=((tf.TensorShape([None]), tf.TensorShape([])), (tf.TensorShape([None]), tf.TensorShape([]))))
    ds = ds.prefetch(1)

    iterator = ds.make_initializable_iterator()
    initializer = iterator.initializer
    ((input_seq, input_len), (target_seq, target_len)) = iterator.get_next()

    return Helper(initializer=initializer, input_seq=input_seq, input_len=input_len, target_seq=target_seq, target_len=target_len, ix_to_char=ix_to_char, char_to_ix=char_to_ix, sos_id=sos_id, eos_id=eos_id, batch_size=batch_size)
        
