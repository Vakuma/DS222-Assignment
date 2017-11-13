from __future__ import print_function

import tensorflow as tf
import sys
import time
import numpy as np
import tensorflow as tf
from collections import Counter



i=0
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss,accuracy_score

import pydoop.hdfs as hdfs

# parameter_servers = ["10.24.1.201:2228"]
# workers = ["10.24.1.32:2228"]
parameter_servers = ["/job:localhost/replica:0/task:0/device:CPU:0"]
workers = ["/job:localhost/replica:0/task:0/device:CPU:0"]



def sigmoid(x):
  return 1 / (1 + np.exp(-x))

vocab = Counter()
labels=Counter()

with hdfs.open('/user/ds222/assignment-1/DBPedia.verysmall/verysmall_train.txt') as f:
  for line in f:
    first,next=line.split(' ',1)
    for label in first.split(','):
      labels[label]+=1
      words = next.strip().lower().split()
      for word in words:
        if(len(word)>=4):
          if(word[0]!='<'):
            vocab[word]+=1
    i=i+1
  #print(i)
#print(counter)


#Convert words to indexes
def get_word_2_index(vocab):
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word] = i
        
    return word2index
#Now we have an index
word2index = get_word_2_index(vocab)
label2index = get_word_2_index(labels)
total_words = len(vocab)
total_labels = len(labels)
total_lines=i
print(total_words)
print(total_labels)
print(total_lines)

X_train=np.zeros((total_lines,total_words))
Y_train=np.zeros((total_lines,total_labels))
i=0
with hdfs.open('/user/ds222/assignment-1/DBPedia.verysmall/verysmall_train.txt') as f:
  for line in f:
    first,next=line.split(' ',1)
    for label in first.split(','):
      Y_train[i,label2index[label]]=1
 
    words = next.strip().lower().split()
    for word in words:
      if(len(word)>=4):
        if(word[0]!='<'):
          X_train[i,word2index[word]]=1
    i=i+1
with hdfs.open('/user/ds222/assignment-1/DBPedia.verysmall/verysmall_test.txt') as f:
  for line in f:
    i=i+1
total_lines_test=i
print(total_lines_test)
X_test=np.zeros((total_lines_test,total_words))
Y_test=np.zeros((total_lines_test,total_labels))
i=0
with hdfs.open('/user/ds222/assignment-1/DBPedia.verysmall/verysmall_test.txt') as f:
  for line in f:
    try:
      first,next=line.split(' ',1)
      for label in first.split(','):
        Y_test[i,label2index[label]]=1
 
      words = next.strip().lower().split()
      for word in words:
        if(len(word)>=4):
          if(word[0]!='<'):
            X_test[i,word2index[word]]=1
      i=i+1
    except:
      print('KeyNotFound')

def get_batch(X,Y,i,batch_size):
    texts = X[i*batch_size:i*batch_size+batch_size]
    categories = Y[i*batch_size:i*batch_size+batch_size]
     
    return texts,categories




batch_size = 100
learning_rate = 1
training_epochs = 20
logs_path = "/home/vadirajk/tf_parameter/1"
n_input = total_words # Words in vocab
n_classes = total_labels        # Categories: graphics, sci.space and baseball


  # Between-graph replication
for worker_device in workers:
  with tf.device(worker_device):

    # count the number of updates

    # input images
    with tf.name_scope('input'):
      # None -> batch size can be any size, 784 -> flattened mnist image
      x = tf.placeholder(tf.float32, shape=[None, n_input], name="x-input")
      # target 10 output classes
      y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="y-input")

    # model parameters will change during training so we use tf.Variable
  for server_device in parameter_servers:
    with tf.device(server_device):
      tf.set_random_seed(1)
      with tf.name_scope("weights"):
        W1 = tf.Variable(tf.random_normal([n_input, n_classes]))

      # bias
      with tf.name_scope("biases"):
        b1 = tf.Variable(tf.zeros([n_classes]))

    # implement model
    with tf.name_scope("softmax"):
      # y is our prediction
      out = tf.add(tf.matmul(x,W1),b1)
      #out = tf.nn.sigmoid(out)
      y = tf.nn.softmax(out)

    # specify cost function
    with tf.name_scope('cross_entropy'):
      # this is our cost
      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))


    # specify optimizer
    with tf.name_scope('train'):
      # optimizer is an "operation" which we can execute in a session
      grad_op = tf.train.GradientDescentOptimizer(learning_rate)
      '''
      rep_op = tf.train.SyncReplicasOptimizer(grad_op,
                                          replicas_to_aggregate=len(workers),
                                          replica_id=FLAGS.task_index,
                                          total_num_replicas=len(workers),
                                          use_locking=True
                                          )
      train_op = rep_op.minimize(cross_entropy, global_step=global_step)
      '''
      train_op = grad_op.minimize(cross_entropy)

    '''
    init_token_op = rep_op.get_init_tokens_op()
    chief_queue_runner = rep_op.get_chief_queue_runner()
    '''

    with tf.name_scope('Accuracy'):
      # accuracy
      correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create a summary for our cost and accuracy
    tf.summary.scalar("cost", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)

    # merge all summaries into a single "operation" which we can
#execute in a session
    summary_op = tf.summary.merge_all()
    init_op = tf.initialize_all_variables()
    print("Variables initialized ...")

    begin_time = time.time()
    frequency = 10
    with tf.Session() as sess:
      init = tf.global_variables_initializer()
      sess.run(init)
      '''
      # is chief
      if FLAGS.task_index == 0:
        sv.start_queue_runners(sess, [chief_queue_runner])
        sess.run(init_token_op)
      '''
      # create log writer object (this will log on every machine)
      writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

      # perform training cycles
      start_time = time.time()
      for epoch in range(training_epochs):

        # number of batches in one epoch
        batch_count = int((total_lines)/batch_size)

        count = 0
        for i in range(batch_count):
          batch_x, batch_y =  get_batch(X_train,Y_train,i,batch_size)

          # perform the operations we defined earlier on batch
          _, cost,acc,summary = sess.run(
                          [train_op, cross_entropy,accuracy, summary_op],
                          feed_dict={x: batch_x, y_: batch_y})
          writer.add_summary(summary)

          count += 1
          if count % frequency == 0 or i+1 == batch_count:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print(" Epoch: %2d," % (epoch+1),
                  " Batch: %3d of %3d," % (i+1, batch_count),
                  " Cost: %.4f," % cost,
                  "Accuracy %.4f," % acc,
                  " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
            count = 0

  #     print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x:
  # mnist.test.images, y_: mnist.test.labels}))
      print("Total Time: %3.2fs" % float(time.time() - begin_time))
      print("Final Cost: %.4f" % cost)

    print("done")
