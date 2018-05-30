import tensorflow as tf
from helper import build_helper
from model import Model 


helper = build_helper()
lipsum = Model(helper=helper, mode='infer')
lipsum.load_model('models/model.ckpt-36420')

lipsum.sess.run(helper.initializer)
while True:
    try:
        input_seq, target_seq, outputs = lipsum.sess.run([lipsum.input_seq, lipsum.target_seq, lipsum.outputs])
        print('Input: {}'.format(''.join(helper.ix_to_char[i] for i in input_seq[0])))
        print('Target: {}'.format(''.join(helper.ix_to_char[i] for i in target_seq[0])))
        print('Prediction: {}'.format(''.join(helper.ix_to_char[i] for i in outputs.sample_id[0])))
        # print(''.join(helper.ix_to_char[i] for i in outputs.sample_id[0]))
        print()
    except tf.errors.OutOfRangeError:
        # lipsum.sess.run(helper.initializer)
        break

