import time
import mxnet as mx
from mxnet import nd

import nn
import data_gen

num_sets = 64
num_assoc = 16

batch_size = 64
num_epochs = 100
num_train_batches = 100
num_test_batches = 10

train_data, train_labels = data_gen.GenData(num_train_batches, 
                                            batch_size, 
                                            num_sets,
                                            speed=2,
                                            coverage=1)

test_data, test_labels = data_gen.GenData(num_test_batches, 
                                          batch_size, 
                                          num_sets,
                                          speed=2,
                                          coverage=1)

net = nn.train(train_data, train_labels, test_data, test_labels, batch_size, num_epochs)
