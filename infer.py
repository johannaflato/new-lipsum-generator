import tensorflow as tf
import numpy as np
from helper import build_helper
from model import Model 


helper = build_helper()
lipsum = Model(helper=helper, mode='infer')
lipsum.load_model('models/model.ckpt-2450')
eos_id = helper.char_to_ix['>']

input_seq = 'On ends of good and evil.>'
input_seq = [helper.char_to_ix[c] for c in input_seq]
new_lipsum = open('new_lipsum.txt', 'w')

for i in range(10000):
    if input_seq[-1] != eos_id: np.append(input_seq, eos_id)
    
    outputs = lipsum.sess.run(lipsum.outputs, feed_dict={ lipsum.input_raw: input_seq })
    output_seq = outputs.sample_id[0]
    text = ''.join(helper.ix_to_char[i] for i in output_seq).strip('>')
    print(i)
    print(text)
    new_lipsum.write(text)
    new_lipsum.write('\n')

    input_seq = output_seq
new_lipsum.close()

