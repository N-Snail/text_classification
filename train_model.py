import tensorflow as tf
from load_data import load_dict, data2idx,loadDataset,getBatches
from base_model import TextRNN
from tqdm import tqdm
import os


tf.app.flags.DEFINE_integer('steps_per_checkpoint', 1000, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_string('model_name', 'classify.ckpt', 'File name used for model checkpoints')
tf.app.flags.DEFINE_integer("num_classes",6,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string("ckpt_dir","text_rnn_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer('rnn_size', 64, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer("vocab_size",40000, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_integer("embed_size",300, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer('numEpochs', 5, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("validate_every", 50, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_integer("dropout_keep_prob", 0.5, " the value of dropout_keep_prob")
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
# tf.app.flags.DEFINE_string("traning_data_path","train-zhihu4-only-title-all.txt","path of traning data.") #train-zhihu4-only-title-all.txt===>training-data/test-zhihu4-only-title.txt--->'training-data/train-zhihu5-only-title-multilabel.txt'
tf.app.flags.DEFINE_string("word2vec_model_path","zhihu-word2vec.bin-100","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string('model_dir', 'test_1/', 'Path to save model checkpoints')
FLAGS = tf.app.flags.FLAGS

word2idx, idx2word = load_dict()
data_path = '../data/test_data/train_new_data_idx.pkl'
trainingSamples = loadDataset(data_path)

# test_path = '../data/test_data_ids.pkl'
# testingSamples = loadDataset(test_path)

model = TextRNN(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size,  FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.vocab_size,
                FLAGS.embed_size,FLAGS.rnn_size, FLAGS.is_training)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters...')
        model.saver.restore(sess,ckpt.model_checkpoint_path)
else:
        print('Created new model parameters...')
        sess.run(tf.global_variables_initializer())

current_step = 0
# loss = 0.0
# previous_losses =[]
# summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)
for e in range(FLAGS.numEpochs):
        print("------Epoch{}/{}------".format(e+1, FLAGS.numEpochs))
        batches = getBatches(trainingSamples,FLAGS.batch_size)
        for nextBatch in tqdm(batches,desc="Training"):
                # learning_rate,step_loss,summary = model.train(sess, nextBatch)
                # print(nextBatch.inputs_sentence)
                feed_dict = {model.input_x: nextBatch.inputs_sentence, model.input_length: nextBatch.inputs_sentence_length,model.input_y: nextBatch.emo_cat, model.dropout_keep_prob: FLAGS.dropout_keep_prob}
                learning_rate,loss, acc, predict, _ = sess.run([model.learning_rate,model.loss_val, model.accuracy, model.predictions, model.train_op], feed_dict=feed_dict)
                current_step+=1
                # loss+=step_loss/100
                if current_step % FLAGS.validate_every == 0:

                        tqdm.write("----- Step %d -- Learning_rate %f -- Loss %.4f -- accuracy %.4f" % (current_step, learning_rate, loss, acc))
                        # summary_writer.add_summary(summary, current_step)
                if current_step % FLAGS.steps_per_checkpoint==0:
                        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                        model.saver.save(sess, checkpoint_path)


