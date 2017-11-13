import pydoop.hdfs as hdfs
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter
i=0
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss,accuracy_score

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
#A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
#v = np.array([1, 0, -1])
#A.dot(v)

nb_epochs=15
batch_size=200
weights=np.random.rand(total_words,total_labels)
avg_loss=[]
avg_acc=[]
avg_loss_test=[]
avg_acc_test=[]
lr=1
epochs=range(nb_epochs)
for i in tqdm(range(nb_epochs)):
	cum_loss=0
	cum_acc=0
	cum_loss_test=0
	cum_acc_test=0
	for j in tqdm(range(total_lines/batch_size)):
		idx=j*(batch_size)
                y_true=Y_train[idx:idx+batch_size,:]
		y_pred=sigmoid(X_train[idx:idx+batch_size].dot(weights))
		weights=weights + lr*( np.transpose(X_train[idx:idx+batch_size,:]).dot(y_true-y_pred))
		cum_loss=cum_loss+log_loss(y_true, y_pred)
		cum_acc=cum_acc+accuracy_score(y_true, np.where(y_pred>=0.5,1,0))
	for j in tqdm(range(total_lines_test/batch_size)):
		idx=j*(batch_size)
                y_true=Y_test[idx:idx+batch_size,:]
		y_pred=sigmoid(X_test[idx:idx+batch_size].dot(weights))
		cum_loss_test=cum_loss+log_loss(y_true, y_pred)
		cum_acc_test=cum_acc+accuracy_score(y_true, np.where(y_pred>=0.5,1,0))
	

	avg_loss+=[cum_loss/(total_lines/batch_size)]
	avg_acc+=[cum_acc/(total_lines/batch_size)]	
	print("Training cross entropy loss is %f" %avg_loss[i])
	print("Training accuracy  is %f" %avg_acc[i])
	avg_loss_test+=[cum_loss_test/(total_lines_test/batch_size)]
	avg_acc_test+=[cum_acc_test/(total_lines_test/batch_size)]	
	print("Test set cross entropy loss is %f" %avg_loss_test[i])
	print("Test set accuracy  is %f" %avg_acc_test[i])
	lr=lr/1


plt.figure(1)
plt.subplot(211)
plt.plot(epochs,avg_loss,'r--',label='Train Loss')
plt.plot(epochs,avg_loss_test,'b--',label='Test Loss')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.subplot(212)
plt.plot(epochs,avg_acc,'r--',label='Train Accuracy')
plt.plot(epochs,avg_acc_test,'b--',label='Test Accuracy')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)	
plt.show()	








