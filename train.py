import tensorflow as tf
from helper import build_helper
from model import Model 

helper = build_helper()
lipsum = Model(helper)

lipsum.sess.run(helper.initializer)
while True:
    try:
        gs, input_seq, target_seq, outputs, loss, _ = lipsum.sess.run([lipsum.global_step, lipsum.input_seq, lipsum.target_seq, lipsum.outputs, lipsum.loss, lipsum.train_op])
        print('Global Step: {}, Loss: {}'.format(gs, loss[0]))
        print('Input: {}'.format(''.join(helper.ix_to_char[i] for i in input_seq[0])))
        print('Target: {}'.format(''.join(helper.ix_to_char[i] for i in target_seq[0])))
        print('Prediction: {}'.format(''.join(helper.ix_to_char[i] for i in outputs.sample_id[0])))
        # print(''.join(helper.ix_to_char[i] for i in outputs.sample_id[0]))
        print()
    except tf.errors.OutOfRangeError:
        lipsum.sess.run(helper.initializer)
