import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from load_data import load_dict, data2idx,loadDataset,getBatches

tf.reset_default_graph()


class TextRNN:
    def __init__(self,num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                 vocab_size,embed_size,rnn_size,is_training,initializer=tf.random_normal_initializer(stddev=0.1)):
        """init all hyperparameter here"""
        # set hyperparamter
        # self.rnn_size=rnn_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        # self.sequence_length=sequence_length  #序列长度
        self.vocab_size=vocab_size  #词典大小
        self.embed_size=embed_size  #向量大小
        self.hidden_size=rnn_size  #隐层大小
        self.is_training=is_training
        self.learning_rate=learning_rate
        self.initializer=initializer
        self.num_sampled=20



        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None,None], name="input_x")
        self.input_length = tf.placeholder(tf.int32,[None],name="input_length")

        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")  # y [None,num_classes]
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")

        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()

        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=15,pad_step_number=True,keep_checkpoint_every_n_hours=1.0)

        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")

        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
    def basis_rnn_cell(self):
        return tf.contrib.rnn.LSTMCell(self.hidden_size)

    def instantiate_weights(self):  #网络权重、偏置定义
        """define all weights here"""
        with tf.name_scope("embedding"):
            # embedding matrix
            self.W_projection = tf.get_variable("W_projection",shape=[self.hidden_size*4, self.num_classes],initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])


    def inference(self):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax """

        wordembedding = np.load('../data/Word_embedding/wordEmbedding300.npy').astype('float32')
        self.Embedding = tf.get_variable("Embedding", dtype=tf.float32, initializer=wordembedding)
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)
        lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size)
        lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size)

        if self.dropout_keep_prob is not None:
            lstm_fw_cell=rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell=rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)

        outputs,outputs_state=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,inputs=self.embedded_words,sequence_length=self.input_length,dtype=tf.float32)

        print("outputs:===>",outputs)
        print("outputs[0]:",outputs[0])
        print("stat ",outputs_state)
        states = []
        for s in outputs_state:
            states.extend([s.h])
        states = tf.concat(states, axis=-1)

        # print("outputs_state:===>",outputs_state)
        #3. concat output
        output_rnn=tf.concat(outputs,axis=2)
        # print(out)
        # self.output_rnn_last=output_rnn[:,-1,:]
        #[batch_size,hidden_size*2] #TODO
        # print("output_rnn_last:", self.output_rnn_last)

        #final_status.c=final_status[0],final_status.h=final_status[-1]
        (fw_outputs, bw_outputs), \
        (fw_final_status, bw_final_status) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                             inputs=self.embedded_words,
                                                                             sequence_length=self.input_length,
                                                                           dtype=tf.float32)
        print("outputs_state:", outputs_state)
        print("fw_outputs:",fw_outputs)
        print("bw_outputs:", bw_outputs)
        print("fw_final_status:", fw_final_status)
        print("fw_final_status.c:", fw_final_status.c)
        print("fw_final_status:", fw_final_status.h)

        c0 = tf.concat(
            [fw_final_status.c, bw_final_status.c], axis=1)
        h0 = tf.concat(
            [fw_final_status.h, bw_final_status.h], axis=1)


        state0 = rnn.LSTMStateTuple(c=c0,h=h0)
        state2 = rnn.LSTMStateTuple(c=outputs_state[0].c,h=outputs_state[0].h)

        state3 = tf.concat([c0,h0],axis=-1)
        # print("final_state0:",final_state)
        print("states0:",state0)


        with tf.name_scope("output"):
            logits = tf.matmul(state3, self.W_projection) + self.b_projection  # [batch_size,num_classes]
        return logits

    def loss(self,l2_lambda=0.0001):
        with tf.name_scope("loss"):

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            # print("labels=self.input_y", self.input_y.get_shape())
            # print("logits=self.logits", self.logits.get_shape())
            # print("logits=self.logits[0]", self.logits[0])
            # print("logits=self.logits[1]", self.logits[1])

            #sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss=tf.reduce_mean(losses)

            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def loss_nce(self,l2_lambda=0.0001): #0.0001-->0.001
        """calculate loss using (NCE)cross entropy here"""
        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        if self.is_training:

            labels=tf.expand_dims(self.input_y,1)
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=tf.transpose(self.W_projection),
                               biases=self.b_projection,
                               labels=labels,
                               inputs=self.output_rnn_last,
                               num_sampled=self.num_sampled,
                               num_classes=self.num_classes,partition_strategy="div"))
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
        return loss



    def train(self):

        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        self.learning_rate=learning_rate
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op

def test123():
    #below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.

    num_classes=6
    learning_rate=0.01
    batch_size=8
    decay_steps=1000
    decay_rate=0.9
    # sequence_length=5
    vocab_size=40000
    embed_size=300
    rnn_size = 64
    is_training=True
    dropout_keep_prob=1#0.5
    textRNN=TextRNN(num_classes, learning_rate, batch_size, decay_steps, decay_rate,vocab_size,embed_size,rnn_size,is_training)

    word2idx, idx2word = load_dict()
    data_path = '../data/test_data/train_new_data_idx.pkl'
    trainingSamples = loadDataset(data_path)

    # test_path = '../data/test_data/test_data_idx.pkl'
    # testingSamples = loadDataset(test_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5):
            batches = getBatches(trainingSamples, 64)
            for e in batches:
                # print("emo_cat_len",len(e.emo_cat))

            # print(batches)
                feed_dict={textRNN.input_x:e.inputs_sentence,
                           textRNN.input_length:e.inputs_sentence_length,
                           textRNN.input_y:e.post_cat,
                           textRNN.dropout_keep_prob:dropout_keep_prob}
            # input_x=np.zeros((batch_size,sequence_length)) #[None, self.sequence_length]
            # input_y=np.array([1,0,1,1,1,2,1,1]) #np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
            # loss,acc,predict,_=sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.train_op],feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob})

                learning_rate, loss, acc, predict, _ = sess.run([textRNN.learning_rate,
                                                                 textRNN.loss_val,
                                                                 textRNN.accuracy,
                                                                 textRNN.predictions,
                                                                 textRNN.train_op],
                                                                feed_dict=feed_dict)
                print("learing rate:",learning_rate,"loss:",loss,"acc:",acc)
                # print(len(predict))
                # print("label:",e.emo_cat)
                # print("predict:",predict)
test123()
