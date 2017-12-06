from __future__ import print_function

import tensorflow as tf
import os
import re
import random


class fetchData(object):
	def __init__(self, ratio = 0.8):
		self.wholedata = []
		self.trainset = []
		self.testset = []
		self.maxSeqLen = 0
		self.uniqueWord = []
		limit = 200
		for filename in os.listdir("res"):
			with open("res/" + filename) as f:
				old_resSeq = ""
				aaseq = []
				for line in f:
					if re.match("^ATOM.*", line):
						if line[21] == "B":
							continue
						resSeq = line[22:26]
						if resSeq != old_resSeq:
							old_resSeq = resSeq
							if len(line[17:20].strip()) == 3:
								aaseq.append(line[17:20])
				self.maxSeqLen = max(len(aaseq), self.maxSeqLen)
				if len(aaseq) > limit:
					aaseq = aaseq[:limit]
				self.wholedata.append(self.dataSet(aaseq, [1., 0.], len(aaseq)))
				self.uniqueWord += aaseq
		for filename in os.listdir("dna"):
			with open("dna/" + filename) as f:
				old_resSeq = ""
				aaseq = []
				for line in f:
					if re.match("^ATOM.*", line):
						if line[21] == "B":
							continue
						resSeq = line[22:26]
						if resSeq != old_resSeq:
							old_resSeq = resSeq
							if len(line[17:20].strip()) == 3:
								aaseq.append(line[17:20])
				self.maxSeqLen = max(len(aaseq), self.maxSeqLen)
				if len(aaseq) > limit:
					aaseq = aaseq[:limit]
				self.wholedata.append(self.dataSet(aaseq, [0., 1.], len(aaseq)))
				self.uniqueWord += aaseq

		random.shuffle(self.wholedata)
		for i in range(int(len(self.wholedata)*ratio)):
			self.trainset.append(self.wholedata[i])
			for _ in range(limit - self.trainset[i].seqlen):
				self.trainset[i].seq.append("NULL")
		for i in range(int(len(self.wholedata)*ratio), len(self.wholedata)):
			self.testset.append(self.wholedata[i])
			for _ in range(limit - self.testset[i-int(len(self.wholedata)*ratio)].seqlen):
				self.testset[i-int(len(self.wholedata)*ratio)].seq.append("NULL")

		# self.trainset = self.wholedata[:int(len(self.wholedata)*ratio)]
		# self.testset = self.wholedata[int(len(self.wholedata)*ratio):]
		self.uniqueWord += ["NULL"]
		self.uniqueWord = list(set(self.uniqueWord))
		self.uniqueWord.sort()
		self.char2id_dict = {w: i for i, w in enumerate(self.uniqueWord)}
		self.id2char_dict = {i: w for i, w in enumerate(self.uniqueWord)}

		self.batch_id = 0
		self.maxSeqLen = limit

	class dataSet(object):
		def __init__(self, seq, label, seqlen):
			self.seq = seq
			self.label = label
			self.seqlen = seqlen

	def train_next(self, batch_size):
		#TODO: finish batch_size implementation
		if self.batch_id == len(self.trainset):
			self.batch_id = 0
		trainsubset = self.trainset[self.batch_id : min(self.batch_id + batch_size, len(self.trainset))]
		batch_data = []
		batch_label = []
		batch_seqlen = []
		for i in trainsubset:
			batch_data.append([[self.char2id(c)] for c in i.seq])
			batch_label.append(i.label)
			batch_seqlen.append(i.seqlen)
		self.batch_id = min(self.batch_id + batch_size, len(self.trainset))
		return batch_data, batch_label, batch_seqlen

	def getTestData(self):
		test_data = []
		test_label = []
		test_seqlen = []
		for i in self.testset:
			test_data.append([[self.char2id(c)] for c in i.seq])
			test_label.append(i.label)
			test_seqlen.append(i.seqlen)
		return test_data, test_label, test_seqlen

	def char2id(self, c):
		return self.char2id_dict[c]

	def id2char(self, id):
		return self.id2char_dict[id]


# Parameters
learning_rate = 0.0001
training_steps = 100000
batch_size = 10
display_step = 100

# Network Parameters
n_hidden = 256 # hidden layer num of features
n_classes = 2 # linear sequence or not

dataSets = fetchData(ratio = 0.8)

# tf Graph input
x = tf.placeholder("float", [None, dataSets.maxSeqLen, 1])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
	'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
	'out': tf.Variable(tf.random_normal([n_classes]))
}

# embedding = tf.Variable(tf.random_normal([n_classes, n_hidden]))
def dynamicRNN(x, seqlen, weights, biases):
    
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
	x = tf.unstack(x, dataSets.maxSeqLen, 1)

	# Define a lstm cell with tensorflow
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

	# inputs = tf.nn.embedding_lookup(embedding, x)
	# Get lstm cell output, providing 'sequence_length' will perform dynamic
	# calculation.
	outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32,
								sequence_length=seqlen)

	# When performing dynamic calculation, we must retrieve the last
	# dynamically computed output, i.e., if a sequence length is 10, we need
	# to retrieve the 10th output.
	# However TensorFlow doesn't support advanced indexing yet, so we build
	# a custom op that for each sample in batch size, get its length and
	# get the corresponding relevant output.

	# 'outputs' is a list of output at every timestep, we pack them in a Tensor
	# and change back dimension to [batch_size, n_step, n_input]
	outputs = tf.stack(outputs)
	outputs = tf.transpose(outputs, [1, 0, 2])

	# Hack to build the indexing and retrieve the right output.
	batch_size = tf.shape(outputs)[0]
	# Start indices for each sample
	index = tf.range(0, batch_size) * dataSets.maxSeqLen + (seqlen - 1)
	# Indexing
	outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

	# Linear activation, using outputs computed above
	return tf.matmul(outputs, weights['out']) + biases['out']

print("Creating RNN")
pred = dynamicRNN(x, seqlen, weights, biases) 
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
	print("run init")
	# Run the initializer
	sess.run(init)

	print("run")
	for step in range(1, training_steps + 1):
		batch_x, batch_y, batch_seqlen = dataSets.train_next(batch_size)
		# print(batch_x)
		# print(batch_y)
		# print(batch_seqlen)
		# Run optimization op (backprop)
		sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
									   seqlen: batch_seqlen})
		if step % display_step == 0 or step == 1:
			# Calculate batch accuracy & loss
			acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,
												seqlen: batch_seqlen})
			print("Step " + str(step*batch_size) + ", Minibatch Loss= " + \
				  "{:.6f}".format(loss) + ", Training Accuracy= " + \
				  "{:.5f}".format(acc))

	print("Optimization Finished!")



	test_data, test_label, test_seqlen = dataSets.getTestData()
	print("Testing Accuracy:", \
		sess.run(accuracy, feed_dict={x: test_data, y: test_label,
									  seqlen: test_seqlen}))
