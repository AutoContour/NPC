import os
os.environ['THEANO_FLAGS']="device=gpu0,floatX=float32"

import sys
import numpy as np
import pickle
import theano
import theano.tensor as T

import lasagne
from lasagne.layers import *
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
from lasagne.nonlinearities import rectify
from lasagne.regularization import regularize_network_params, regularize_layer_params, l1, l2

from three_channel_dicomSubject import *
import skimage
import skimage.segmentation
import medpy.metric
import codecs



class LogisticRegression(object):
    def __init__(self, input_feature):
        self.batch_size, self.n_class, self.dim_x, self.dim_y, self.dim_z = input_feature.shape
        self.input = input_feature.dimshuffle(0,2,3,4,1).reshape((self.batch_size*self.dim_x*self.dim_y*self.dim_z, self.n_class))
        self.p_y_given_x = T.nnet.softmax(self.input)
        self.score_map = self.p_y_given_x.reshape((self.batch_size, self.dim_x, self.dim_y, self.dim_z, self.n_class))[:,:,:,:,1]

    def negative_log_likelihood(self, label):
        y = label.reshape((self.batch_size*self.dim_x*self.dim_y*self.dim_z,))
        loss = -T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]

        mask = y * 10.
        weighted_loss = T.mean(loss + loss * mask)

        return weighted_loss







def build_res_V1(input_var, batch_size):
    
    net = {}

    net['input'] = InputLayer((batch_size, 3, None, None, None), input_var=input_var)   ##### 4 change to 3 (3 channel)
    net['conv1a'] = batch_norm(Conv3DDNNLayer(net['input'], 64, (3,3,3), pad='same', nonlinearity=rectify))
    net['conv1b'] = batch_norm(Conv3DDNNLayer(net['conv1a'], 64, (3,3,3), pad='same', nonlinearity=rectify))

    net['conv1c'] = Conv3DDNNLayer(net['conv1b'], num_filters=64, filter_size=(3,3,3), stride=(2,2,2), pad='same', nonlinearity=None)
    net['pool1'] = MaxPool3DDNNLayer(net['conv1b'], pool_size=(2,2,2)) # 80,80,16

    # Residual 2
    net['res2'] = BatchNormLayer(net['conv1c'])
    net['res2'] = NonlinearityLayer(net['res2'], nonlinearity=rectify)
    net['res2'] = batch_norm(Conv3DDNNLayer(net['res2'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=rectify))
    net['res2'] = Conv3DDNNLayer(net['res2'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=None)
    net['res2'] = ElemwiseSumLayer([net['res2'], net['conv1c']])

    # Residual 3
    net['res3'] = BatchNormLayer(net['res2'])
    net['res3'] = NonlinearityLayer(net['res3'], nonlinearity=rectify)
    net['res3'] = batch_norm(Conv3DDNNLayer(net['res3'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=rectify))
    net['res3'] = Conv3DDNNLayer(net['res3'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=None)
    net['res3'] = ElemwiseSumLayer([net['res3'], net['res2']])

    net['bn3'] = BatchNormLayer(net['res3'])
    net['relu3'] = NonlinearityLayer(net['bn3'], nonlinearity=rectify)

    net['conv3a'] = Conv3DDNNLayer(net['relu3'], num_filters=64, filter_size=(3,3,3), stride=(2,2,1), pad='same', nonlinearity=None)
    net['pool2'] = MaxPool3DDNNLayer(net['relu3'], pool_size=(2,2,1)) # 40,40,16

    # Residual 4
    net['res4'] = BatchNormLayer(net['conv3a'])
    net['res4'] = NonlinearityLayer(net['res4'], nonlinearity=rectify)
    net['res4'] = batch_norm(Conv3DDNNLayer(net['res4'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=rectify))
    net['res4'] = Conv3DDNNLayer(net['res4'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=None)
    net['res4'] = ElemwiseSumLayer([net['res4'], net['conv3a']])

    # Residual 5
    net['res5'] = BatchNormLayer(net['res4'])
    net['res5'] = NonlinearityLayer(net['res5'], nonlinearity=rectify)
    net['res5'] = batch_norm(Conv3DDNNLayer(net['res5'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=rectify))
    net['res5'] = Conv3DDNNLayer(net['res5'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=None)
    net['res5'] = ElemwiseSumLayer([net['res5'], net['res4']])

    net['bn5'] = BatchNormLayer(net['res5'])
    net['relu5'] = NonlinearityLayer(net['bn5'], nonlinearity=rectify)
    net['conv5a'] = Conv3DDNNLayer(net['relu5'], num_filters=64, filter_size=(3,3,3), stride=(2,2,2), pad='same', nonlinearity=None)

    # Residual 6
    net['res6'] = BatchNormLayer(net['conv5a'])
    net['res6'] = NonlinearityLayer(net['res6'], nonlinearity=rectify)
    net['res6'] = batch_norm(Conv3DDNNLayer(net['res6'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=rectify))
    net['res6'] = Conv3DDNNLayer(net['res6'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=None)
    net['res6'] = ElemwiseSumLayer([net['res6'], net['conv5a']])

    # Residual 7
    net['res7'] = BatchNormLayer(net['res6'])
    net['res7'] = NonlinearityLayer(net['res7'], nonlinearity=rectify)
    net['res7'] = batch_norm(Conv3DDNNLayer(net['res7'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=rectify))
    net['res7'] = Conv3DDNNLayer(net['res7'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=None)
    net['res7'] = ElemwiseSumLayer([net['res7'], net['res6']])

    net['bn7'] = BatchNormLayer(net['res7'])
    net['relu7'] = NonlinearityLayer(net['bn7'], nonlinearity=rectify)

    net['conv8'] = batch_norm(Conv3DDNNLayer(net['relu7'], num_filters=64, filter_size=(3,3,3), pad='same', nonlinearity=rectify))

    # upscale 1
    net['upscale1'] = Upscale3DLayer(net['conv8'], scale_factor=(2,2,2), mode='repeat')
    net['concat1'] = ConcatLayer([net['pool2'], net['upscale1']])
    net['upconv1a'] = batch_norm(Conv3DDNNLayer(net['concat1'], 64, (1,1,1), pad='same', nonlinearity=rectify))
    net['upconv1b'] = batch_norm(Conv3DDNNLayer(net['upconv1a'], 64, (3,3,3), pad='same', nonlinearity=rectify))

    # upscale 2
    net['upscale2'] = Upscale3DLayer(net['upconv1b'], scale_factor=(2,2,1), mode='repeat')
    net['concat2'] = ConcatLayer([net['pool1'], net['upscale2']])
    net['upconv2a'] = batch_norm(Conv3DDNNLayer(net['concat2'], 64, (1,1,1), pad='same', nonlinearity=rectify))
    net['upconv2b'] = batch_norm(Conv3DDNNLayer(net['upconv2a'], 64, (3,3,3), pad='same', nonlinearity=rectify))

    # upscale 3
    net['upscale3'] = Upscale3DLayer(net['upconv2b'], scale_factor=(2,2,2), mode='repeat')
    net['upconv3a'] = batch_norm(Conv3DDNNLayer(net['upscale3'], 64, (1,1,1), pad='same', nonlinearity=rectify))
    net['upconv3b'] = batch_norm(Conv3DDNNLayer(net['upconv3a'], 64, (3,3,3), pad='same', nonlinearity=rectify))

    net['output'] = batch_norm(Conv3DDNNLayer(net['upconv3b'], 2, (3,3,3), pad='same', nonlinearity=None))

    params = lasagne.layers.get_all_params(net['output'], trainable=True)
    l2_penalty = regularize_network_params(net['output'], l2)

    return net, params, l2_penalty



def train_model_res_V1(results_path, fine_tune=False, batch_size=5, base_lr=0.001, n_epochs=30):

    ftensor5 = T.TensorType('float32', (False,)*5)
    x = ftensor5()
    y = T.itensor4('y')

    network, params, l2_penalty = build_res_V1(x, batch_size)
    
    train_cost = []
    
    if fine_tune is True: # Fine tune the model if this flag is True
        with np.load(os.path.join(results_path,'params.npz')) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            set_all_param_values(network['output'], param_values[0])
            print 'initialization done!'

    prediction = get_output(network['output'])
    loss_layer = LogisticRegression(prediction)
    cost_output = loss_layer.negative_log_likelihood(y)

    lamda=0.0001
    cost = cost_output + lamda * l2_penalty
    updates = lasagne.updates.adadelta(cost, params)
    train = theano.function([x, y], [cost, cost_output], updates=updates)

    print 'function graph done!'

    itr = 0
    test_min = np.inf
    train_cost = []

    data_folder = '/DATA/PATH'
    file_name = results_path + "/log_loss.txt"
    fw = codecs.open(file_name, "w", "utf-8-sig")
    for train_x, train_y in load_train_negative(batch_size=batch_size, n_epochs=n_epochs, patchSize=[48,48,16]):
        print 'train_x shape: {}, positive percentage: {}'.format(train_x.shape, np.mean(train_y))
        n_train_batches = train_x.shape[0] / batch_size
        for minibatch_index in xrange(n_train_batches):
            
            train_x_itr = train_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size,:,:,:]
            train_y_itr = train_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size,:,:,:]
            train_cost_itr, train_cost_itr_classify = train(train_x_itr, train_y_itr)

            train_cost.append([train_cost_itr,train_cost_itr_classify])

            print 'model: {}, itr: {}, train loss overall: {}, train loss classify: {}'.format('resV1', itr, train_cost_itr, train_cost_itr_classify)
            print >> fw, 'model: {}, itr: {}, train loss overall: {}, train loss classify: {}'.format('resV1', itr, train_cost_itr, train_cost_itr_classify)
            itr = itr + 1


        if itr % 200 == 0:
            np.savez(os.path.join(results_path, 'params_'+str(itr)+'.npz'), get_all_param_values(network['output']))
            print 'save model done ...'
    fw.close()



if __name__ == '__main__':


    results_path = '/PATH/TO/SAVE/MODEL'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        print 'make folder', results_path

    train_model_res_V1(results_path=results_path, fine_tune=False)


