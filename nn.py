from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon

data_ctx = mx.cpu()
model_ctx = mx.cpu()

def evaluate_accuracy(data_iterator, net):
#    num_access = 0
#    num_predict = 0
#    softmax_cross_entropy = gluon.loss.L1Loss()
#        output = nd.rint(nd.maximum(0, output))
#        accuracies.append(nd.mean(nd.abs(output-label)).asscalar())
#        loss = softmax_cross_entropy(output, label)
#        accuracies.append(nd.sum(loss).asscalar())
#    return sum(accuracies)/float(len(accuracies))
#        num_access += nd.sum(label).asscalar()
#        num_access += nd.sum(nd.maximum(0, output)).asscalar()
#        num_predict += nd.sum(nd.minimum(label, output)).asscalar()
#    return 0 if num_access == 0 else num_predict*1.0/num_access

    accuracies = []
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        output = net(data)

        batch_size = data.shape[0]
        num_sets = data.shape[1]
        num_access = 0
        for j in range(num_sets):
            if label[0][j].asscalar() > 0:
                num_access += 1

        predict_index = nd.topk(output, k=num_access)
        predict_index = nd.sort(predict_index, axis=1)
        correct = 0.0
        for j in range(batch_size):
            for k in range(num_access):
                if label[j][int(predict_index[j][k].asscalar())].asscalar() == k+1:
                    correct += 1.0

        accuracies.append(correct/batch_size/num_access)

    return sum(accuracies)/len(accuracies)


def train(train_data, train_labels, test_data, test_labels, batch_size, num_epochs):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(train_labels.shape[2], activation="relu"))
        net.add(gluon.nn.Dense(train_labels.shape[2], activation="relu"))
        net.add(gluon.nn.Dense(train_labels.shape[2]))

    net.collect_params().initialize(mx.init.Normal(sigma=0.1), ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
#    softmax_cross_entropy = gluon.loss.L1Loss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

    for e in range(num_epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(zip(train_data, train_labels)):
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()

        train_accuracy = evaluate_accuracy(zip(train_data, train_labels), net)
        test_accuracy = evaluate_accuracy(zip(test_data, test_labels), net)
        print("%s %s %s" % (cumulative_loss, train_accuracy, test_accuracy))

    return net
