import tensorflow as tf
from helper import build_helper
from model import Model 


helper = build_helper()
lipsum = Model(helper=helper, mode='train')
lipsum.load_model('models/model.ckpt-36420')

lipsum.sess.run(helper.initializer)
epoch = 0
total_epochs = 15
epoch_loss = 0
while epoch < total_epochs:
    try:
        gs, input_seq, target_seq, outputs, loss, _ = lipsum.sess.run([lipsum.global_step, lipsum.input_seq, lipsum.target_seq, lipsum.outputs, lipsum.loss, lipsum.train_op])
        # print('Global Step: {}, Loss: {}'.format(gs, loss[0]))
        # print('Input: {}'.format(''.join(helper.ix_to_char[i] for i in input_seq[0])))
        # print('Target: {}'.format(''.join(helper.ix_to_char[i] for i in target_seq[0])))
        # print('Prediction: {}'.format(''.join(helper.ix_to_char[i] for i in outputs.sample_id[0])))
        # print(''.join(helper.ix_to_char[i] for i in outputs.sample_id[0]))
        # print()
        epoch_loss += loss
    except tf.errors.OutOfRangeError:
        print(epoch_loss)
        epoch_loss = 0
        epoch += 1
        lipsum.sess.run(helper.initializer)

        if epoch % 5 == 0:
            lipsum.save_model()



