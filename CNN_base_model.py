import tensorflow as tf
import numpy as np

from load_data import load_dict, data2idx,loadDataset,getBatches

class TextCNN:
    def __init__(self, filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size
                 ,initializers=tf.random_normal_initializer(stddev=0.1),is_training_flag=True,clip_gradients=5.0,decay_rate_big=0.50):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")#ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes=filter_sizes # it is a list of int. e.g. [3,4,5]
        self.num_filters=num_filters
        self.initializers=initializers
        self.num_filters_total=self.num_filters * len(filter_sizes) #how many filters totally.

        self.clip_gradients = clip_gradients


        # add placeholder (X,label)
        self.is_training_flag = tf.placeholder(tf.bool, name="is_training_flag")
        self.input_length = tf.placeholder(tf.int32, [None], name="input_length")

        self.input_x = tf.placeholder(tf.int32, [None,None], name="input_x")

        self.input_y = tf.placeholder(tf.int32, [None],name="input_y")  # y:[None,num_classes]

        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")

        self.iter = tf.placeholder(tf.int32) #training iteration
        self.tst=tf.placeholder(tf.bool)
        self.use_mulitple_layer_cnn=False

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))

        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference() #[None, self.label_size]. main computation graph is here
        self.possibility=tf.nn.sigmoid(self.logits)
        self.loss_val = self.loss()
        self.train_op = self.train()


        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]

        # print("self.predictions:", self.predictions)
        # print("self.predictions:",self.predictions.get_shape())
        # print(tf.argmax(self.input_y, 1).get_shape())
        # correct_prediction = tf.equal(tf.cast(self.predictions, tf.int64), tf.argmax(self.input_y, 1))
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) #tf.argmax(self.logits, 1)-->[batch_size]
        # print("self.input_y:",self.input_y.get_shape())
        self.accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()



    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"): # embedding matrix
            # self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            wordembedding = np.load('../data/Word_embedding/wordEmbedding300.npy').astype('float32')
            self.Embedding = tf.get_variable("Embedding", dtype=tf.float32, initializer=wordembedding)
            self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.num_classes],initializer=self.initializers)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])  # [label_size] #ADD 2017.06.09

    def inference(self):
        """main computation graph here: 1.embedding-->2.CONV-BN-RELU-MAX_POOLING-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)#[None,sentence_length,embed_size]
        self.sentence_embeddings_expanded=tf.expand_dims(self.embedded_words,-1)
        #[None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;
        # tf.nn.relu;tf.nn.max_pool;
        # feature shape is 4-d. feature is a new variable
        #if self.use_mulitple_layer_cnn: # this may take 50G memory.
        #    print("use multiple layer CNN")
        #    h=self.cnn_multiple_layers()
        #else: # this take small memory, less than 2G memory.
        print("use single layer CNN")
        h=self.cnn_single_layer()
        #5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(h,self.W_projection) + self.b_projection  #shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
            # print("logits:",logits.get_shape())
            print("logits:",logits)

        return logits


    # def cnn_single_layer(self):
    #     pooled_outputs = []
    #     for i, filter_size in enumerate(self.filter_sizes):
    #         with tf.variable_scope("convolution-pooling-%s" % filter_size):
    #             # ====>a.create filter
    #             #[ filter_height, filter_weight, in_channel, out_channels ]
    #             filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],initializer=tf.random_normal_initializer())
    #
    #             # ====>b.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
    #             # Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
    #             # Conv.Returns: A `Tensor`. Has the same type as `input`.
    #             #         A 4-D tensor. The dimension order is determined by the value of `data_format`, see below for details.
    #             # 1)each filter with conv2d's output a shape:[1,sequence_length-filter_size+1,1,1];2)*num_filters--->[1,sequence_length-filter_size+1,1,num_filters];3)*batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]
    #             # input data format:NHWC:[batch, height, width(embe), channels]
    #             # print(self.sentence_embeddings_expanded.get_shape())
    #             #self.sentence_embeddings_expanded,shape[weight=embe,seqence_length]
    #             conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",name="conv")
    #             # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
    #             # print("self.sentence_embeddings_expanded:",self.sentence_embeddings_expanded)
    #             # print("conv:",conv.get_shape())
    #             # print(conv)
    #
    #             conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn_bn_')
    #             # print(conv.get_shape())
    #             # ====>c. apply nolinearity
    #             b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
    #             h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")
    #
    #             # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
    #             # ====>. max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
    #             #                  ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
    #             #                  strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
    #             # pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],strides=[1, 1, 1, 1], padding='VALID',name="pool")  # shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
    #
    #             pooling_max = tf.squeeze(tf.nn.max_pool(h, ksize=[1, self.sequence_length, 1, 1], strides=[1, 1, 1, 1], padding='VALID',name="pool"))
    #             # pooling_avg=tf.squeeze(tf.reduce_mean(h,axis=1)) #[batch_size,num_filters]
    #             # print(i, "pooling:", pooling_max.get_shape())
    #             # pooling=tf.concat([pooling_max,pooling_avg],axis=1) #[batch_size,num_filters*2]
    #             # pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],strides=[1, 1, 1, 1], padding='VALID', name="pool")
    #             # pooled_outputs.append(pooled)#[batch_size,embed,(height)1,(filter_num)2]
    #             pooled_outputs.append(pooling_max)  # h:[batch_size,sequence_length,1,num_filters]
    #             # print("pooled:",pooled)
    #             # print("pooled_outputs:",pooled_outputs)
    #     # 3.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
    #
    #     self.h_pool = tf.concat(pooled_outputs,axis=1)
    #     # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
    #     # print("h_pool:",self.h_pool.get_shape())
    #
    #     # self.h_pool_flat = tf.reshape(self.h_pool, [-1,self.num_filters_total])
    #     # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)
    #     # print("h_pool_flat:",self.h_pool_flat.get_shape())
    #
    #     # 4.=====>add dropout: use tf.nn.dropout
    #     with tf.name_scope("dropout"):
    #         # self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]
    #         h = tf.nn.dropout(self.h_pool,
    #                           keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]
    #     # h = tf.layers.dense(self.h_drop, self.num_filters_total, activation=tf.nn.tanh, use_bias=True)
    #     # shape: [batch_size, 1, 1, num_filters_total]
    #     # a= np.array(h)
    #     print("h_pool:",self.h_pool)
    #     print("h:",h.get_shape)
    #     return h  # [batch_size,sequence_length - filter_size + 1,num_filters]

    def cnn_single_layer(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("convolution-pooling-%s" % filter_size):
                print("self.sentence_embeddings_expanded:",self.sentence_embeddings_expanded)
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],initializer=tf.random_normal_initializer())
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",name="conv")
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn_bn_')
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")
                pooling_max = tf.squeeze(tf.nn.max_pool(h, ksize=[1, self.sequence_length, 1, 1], strides=[1, 1, 1, 1], padding='VALID',name="pool"))
                pooled_outputs.append(pooling_max)
        self.h_pool = tf.concat(pooled_outputs,axis=1)
        print("self.h_pool:",self.h_pool)
        with tf.name_scope("dropout"):
            h = tf.nn.dropout(self.h_pool,keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]

        print("h_pool:", self.h_pool)
        a=np.array(h)
        print("h:", tf.shape(h))

        return h


    def loss(self,l2_lambda=0.0001):#0.001
        with tf.name_scope("loss"):
            #input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)

            #sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss=tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss


    # def loss(self,l2_lambda=0.0001): #0.0001#this loss function is for multi-label classification
    #     with tf.name_scope("loss"):
    #         #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
    #         #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
    #         #input_y:shape=(?, 1999); logits:shape=(?, 1999)
    #         # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    #         losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits);#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
    #         #losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
    #         print("sigmoid_cross_entropy_with_logits.losses:",losses) #shape=(?, 1999).
    #         losses=tf.reduce_sum(losses,axis=1) #shape=(?,). loss for all data in the batch
    #         loss=tf.reduce_mean(losses)         #shape=().   average loss in the batch
    #         l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
    #         loss=loss+l2_losses
    #     return loss

    def train_old(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #ADD 2018.06.01
        with tf.control_dependencies(update_ops):  #ADD 2018.06.01
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op


def newtest():

    num_classes=6
    learning_rate=0.001
    batch_size=8
    decay_steps=1000
    decay_rate=0.95
    sequence_length=16
    vocab_size=40000
    embed_size=300
    is_training=True
    dropout_keep_prob=1.0 #0.5
    filter_sizes=[2,3]
    num_filters=2


    textRNN = TextCNN(filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                      sequence_length, vocab_size, embed_size)
    data_path = '../data/test_data/train_new_data_idx.pkl'
    trainingSamples = loadDataset(data_path)

    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       for i in range(500):
           batches = getBatches(trainingSamples,8)
           for e in batches:
               print(" e.inputs_sentence:", e.inputs_sentence)
               print("e.inputs_sentence_length:",e.inputs_sentence_length)
               print("e.em_cat:",e.emo_cat)
               # print("e.emo_cat:",e.emo_cat)
               feed_dict = {textRNN.input_x: e.inputs_sentence,
                            textRNN.input_length: e.cnn_batch_length,
                            textRNN.input_y: e.emo_cat,
                            textRNN.dropout_keep_prob: dropout_keep_prob}
               loss,possibility,accuracy,_=sess.run([textRNN.loss_val,textRNN.possibility,textRNN.accuracy,textRNN.train_op],
                                                    feed_dict={textRNN.input_x:e.inputs_sentence,
                                                               textRNN.input_y:e.emo_cat,
                                                               textRNN.input_length:e.inputs_sentence_length,
                                                               textRNN.dropout_keep_prob:dropout_keep_prob,textRNN.tst:False,textRNN.is_training_flag:is_training})
               print(i,"loss:",loss,"-------------------------------------------------------")
               print("label:",e.emo_cat)
               #print("possibility:",possibility)
               print("acc:",accuracy)


newtest()
