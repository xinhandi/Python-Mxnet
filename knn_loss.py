# Copyright (c) 2017 by Contributors
# \file knn_loss.py
# \author deepearthgo
# \it's just an mxnet-python implementation of CVPR_2017_poster[Reliable Crowdsouring and Deep Localoty-Preserving Learning for Experession Recognition in the Wild]
import os
import sklearn.metrics.pairwise as distance_measure
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '2'
import mxnet as mx
import numpy as np
# define metric of accuracy
class Knn_Accuracy(mx.metric.EvalMetric):
    def __init__(self, num=None):
        super(Knn_Accuracy, self).__init__('Knn_accuracy', num)
        self.num = num

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        if self.num is not None:
            assert len(labels) == self.num

        pred_label = mx.nd.argmax_channel(preds[0]).asnumpy().astype('int32')
        label = labels[0].asnumpy().astype('int32')

        mx.metric.check_label_shapes(label, pred_label)

        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)

# define some metric of center_loss
class KnnLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(KnnLossMetric, self).__init__('knn_loss')

    def update(self, labels, preds):
        self.sum_metric += preds[1].asnumpy()[0]
        self.num_inst += 1

# see details:
# <A Discriminative Feature Learning Approach for Deep Face Recogfnition>
class KnnLoss(mx.operator.CustomOp):
    def __init__(self, ctx, shapes, dtypes, lamna, k_num):
        if not len(shapes[0]) == 2:
            raise ValueError('dim for input_data shoudl be 2 for CenterLoss')

        self.lamna = lamna
        self.batch_size = shapes[0][0]
        #self.num_class = num_class
        #self.scale = scale
        self.k_num = k_num

    def forward(self, is_train, req, in_data, out_data, aux):
        #imd -- image_data [batch_size,n_features]
        #sbm -- similarity batch matrix of imd [batch_size,batch_size]
        #fl  -- full label of sort sbm [batch_size,batch_size] [ascend]
        #mfl -- min k label [batch_size, k]
        #kls -- knn-loss [1]
        #knc -- knn-centers [batch_size,1]
        #dls -- grad of knn-loss [batch_size,1]
        imd = in_data[0].asnumpy() #[n,f]
        knc = aux[0]
        dls = aux[1]
        kls = aux[2]
        kls[:]=0
        #calculate min_k_center
        sbm = distance_measure(imd,imd,metric='euclidean')
        fl = np.argsort(sbm)
        mfl = fl[:,self.k_num+1]#[n,k]

        kls = 0

        for i in range(self.batch_size):

            kls += np.sum(sbm[i,mfl[i,:]])
            knc[i] = (np.sum(imd[mfl[i,:],:]) - imd[i,:])/self.k_num
            dls[i] = imd[i,:]-knc[i]

        self.assign(out_data[0],req[0],mx.nd.array(kls))

        #center_grad =
        #labels = in_data[1].asnumpy()
        #diff = aux[0]
        #center = aux[1]

        # store x_i - c_yi
        #for i in range(self.batch_size):
        #    diff[i] = in_data[0][i] - center[int(labels[i])]

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        #knc = aux[0]
        dls = aux[1]
        #kls = aux[2]

        # back grad is just scale * ( x_i - c_yi)
        # grad_scale = float(self.scale/self.batch_size)
        self.assign(in_grad[0], req[0], mx.nd.array(dls * self.lamna))

        # update the center
        # labels = in_data[1].asnumpy()
        # label_occur = dict()
        # for i, label in enumerate(labels):
        #    label_occur.setdefault(int(label), []).append(i)

        # for label, sample_index in label_occur.items():
        #    sum_[:] = 0
        #    for i in sample_index:
        #        sum_ = sum_ + diff[i]
        #    delta_c = sum_ / (1 + len(sample_index))
        #    center[label] += self.alpha * delta_c

@mx.operator.register("knnloss")
class KnnLossProp(mx.operator.CustomOpProp):
    def __init__(self, lamna, batchsize=64):
        super(KnnLossProp, self).__init__(need_top_grad=False)

        # convert it to numbers
        # self.num_class = int(num_class)
        self.lamna = float(lamna)
        # self.scale = float(scale)
        self.batchsize = int(batchsize)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def list_auxiliary_states(self):
        # call them 'bias' for zero initialization
        return ['knc_bias', 'dls_bias', 'kls_bias']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)

        # store grad , same shape as input batch
        dls_shape = [self.batchsize, data_shape[1]]

        # store the center of each point , same shape as input batch
        knc_shape = [self.batchsize, dls_shape[1]]

        # center loss
        kls_shape = [1, ]

        output_shape = [1, ]
        return [data_shape, label_shape], [output_shape], [dls_shape, knc_shape, kls_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return KnnLoss(ctx, shapes, dtypes, self.lamna, self.k_num)
