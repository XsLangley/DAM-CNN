import tensorflow as tf
import numpy as np
import os
import util
import time


class DAMCNN(object):
    def __init__(self, infor_dict, vgg_path=None):
        # load the trained VGG-FACE model
        if vgg_path is not None:
            self.par_dict = np.load(vgg_path).item()
        else:
            self.par_dict = np.load('vgg_origin_dict.npy').item()

        # initializer flag
        self.iniflag = False

        self.datainfo_dict = infor_dict

    def build(self, attention=True):
        # buffers to store validation information
        self.tt_acc_mat = []
        self.val_acc_mat = []
        # build the computing graph of DAM-CNN
        self.x_in = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input_data')
        self.y_in = tf.placeholder(tf.float32, [None, self.datainfo_dict['cls_num']], name='ground_truth')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_rate_main')
        self.keep_prob_MPAE = tf.placeholder(tf.float32, name='dropout_rate_MPAE')


        # placeholders for expected decoding(reconstructed) targets
        self.x_exp = []
        for i in range(0, self.datainfo_dict['n_dom']):
            self.x_exp.append(tf.placeholder(tf.float32, [None, 7 * 7 * 512], name='GT_decoded_feature_' + str(i)))

        # MPAE
        self.AEs = []

        # weights list of the backbone network
        self.main_var_list = []
        # weights list of the decoders
        self.sub_var_list = []

        # the backbone graph of DAM-CNN: ConvNet+SERD+MPVS-Net
        with tf.device('/gpu:0'):
            with tf.name_scope('conv1'):
                self.conv1_1 = self.conv(self.x_in, 'conv1_1')
                self.conv1_2 = self.conv(self.conv1_1, 'conv1_2')
                self.pool1 = self.pooling(self.conv1_2, 'pooling1')
            with tf.name_scope('conv2'):
                self.conv2_1 = self.conv(self.pool1, 'conv2_1')
                self.conv2_2 = self.conv(self.conv2_1, 'conv2_2')
                self.pool2 = self.pooling(self.conv2_2, 'pooling2')
            with tf.name_scope('conv3'):
                self.conv3_1 = self.conv(self.pool2, 'conv3_1')
                self.conv3_2 = self.conv(self.conv3_1, 'conv3_2')
                self.conv3_3 = self.conv(self.conv3_2, 'conv3_3')
                self.pool3 = self.pooling(self.conv3_3, 'pooling3')
            with tf.name_scope('conv4'):
                self.conv4_1 = self.conv(self.pool3, 'conv4_1')
                self.conv4_2 = self.conv(self.conv4_1, 'conv4_2')
                self.conv4_3 = self.conv(self.conv4_2, 'conv4_3')
                self.pool4 = self.pooling(self.conv4_3, 'pooling4')
            with tf.name_scope('conv5'):
                self.conv5_1 = self.conv(self.pool4, 'conv5_1')
                self.conv5_2 = self.conv(self.conv5_1, 'conv5_2')
                self.conv5_3 = self.conv(self.conv5_2, 'conv5_3')
                self.pool5 = self.pooling(self.conv5_3, 'pooling5')

            with tf.name_scope('att'):
                self.att_mask = self.attention(self.pool5, 'att')
                self.att_mask = tf.reshape(self.att_mask, [-1, 7, 7, 1])
                self.masked_conv = tf.nn.relu(self.pool5 * tf.tile(self.att_mask, [1, 1, 1, 512]))
            with tf.name_scope('fc'):
                if attention:
                    self.fc6 = self.conv(self.masked_conv, 'fc6', padding=False, dropout=True)
                else:
                    self.fc6 = self.conv(self.pool5, 'fc6', padding=False, dropout=True)

                self.fc7 = self.conv(self.fc6, 'fc7', padding=False, dropout=True)
                self.fc8 = self.fully_con(self.fc7, 'fc8')
            with tf.name_scope('output'):
                self.result = tf.nn.softmax(self.fc8, name='prediction')

        # decoders of MPAE
        with tf.device('/gpu:0'):
            with tf.name_scope('dec'):
                for d in range(0, self.datainfo_dict['n_dom']):
                    # instantiate a decoder
                    self.AEs.append(AE(self.fc6, self.keep_prob_MPAE, d))
                    # append the variables to the parameter list
                    self.sub_var_list.append(self.AEs[d].var_list)

    def conv(self, in_data, name, padding=True, dropout=False):
        # image convolution
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='kernel',
                                     initializer=tf.constant(self.par_dict[name]['kernel']))
            bias = tf.get_variable(name='bias',
                                   initializer=tf.constant(self.par_dict[name]['bias']))


        self.main_var_list.append(kernel)
        self.main_var_list.append(bias)

        if padding:
            conv = tf.nn.conv2d(in_data, kernel, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv = tf.nn.conv2d(in_data, kernel, strides=[1, 1, 1, 1], padding='VALID')

        conv_bias = tf.nn.bias_add(conv, bias)
        conv_bias_relu = tf.nn.relu(conv_bias)

        if dropout == True:
            return tf.nn.dropout(conv_bias_relu, keep_prob=self.keep_prob)
        else:
            return conv_bias_relu

    def pooling(self, in_data, name):
        # max-pooling
        return tf.nn.max_pool(in_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def fully_con(self, in_data, name):
        # fully-connected layer (the last layer)

        # flatten the input
        in_data = tf.reshape(in_data, [-1, 4096])
        with tf.variable_scope(name):
            fc8w = tf.get_variable(name='kernel',
                                   initializer=tf.truncated_normal([4096, self.datainfo_dict['cls_num']], stddev=0.1))
            fc8b = tf.get_variable(name='bias',
                                   initializer=tf.constant(0, dtype=tf.float32, shape=[self.datainfo_dict['cls_num']]))

        self.main_var_list.append(fc8w)
        self.main_var_list.append(fc8b)
        return tf.nn.bias_add(tf.matmul(in_data, fc8w), fc8b)

    def attention(self, in_data, name):
        # compute the attention mask
        # input size: BSx7x7x512, kernel size: 1x1x512x1, output size: BSx7x7x1

        with tf.variable_scope(name):
            if 'att' in self.par_dict.keys():
                kernel = tf.get_variable(name='kernel',
                                         initializer=tf.constant(self.par_dict['att']['kernel']))
                bias = tf.get_variable(name='bias',
                                       initializer=tf.constant(self.par_dict['att']['bias']))
            else:
                kernel = tf.get_variable(name='kernel',
                                         initializer=tf.truncated_normal([1, 1, 512, 1], stddev=0.1))
                bias = tf.get_variable(name='bias',
                                       initializer=tf.constant(np.zeros([1]).reshape(-1), dtype=tf.float32))
        att = tf.nn.conv2d(in_data, kernel, strides=[1, 1, 1, 1], padding='VALID')
        att_mask = tf.nn.tanh(tf.nn.bias_add(att, bias))
        self.main_var_list.append(kernel)
        self.main_var_list.append(bias)
        return att_mask

    def train(self, sess, data_path, epoch=10, reg_1=1e-3, reg_2=5*1e-4, lr_main=1e-5, lr_sub=1e-4, reg_ae=1e-5, bs=32):
        # train the vgg network

        # accuracy threshold to save the model
        thr_acc = 0.6

        # classification loss
        self.cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc8, labels=self.y_in))
        # attention regularization of SERD
        self.reg_term = reg_1 * tf.reduce_sum(tf.abs(self.att_mask)) + \
                        reg_2 * tf.reduce_sum(1 - (self.att_mask * self.att_mask))
        self.loss = self.cross_entropy + self.reg_term

        # initialize optimizers
        optimizer_main = tf.train.AdamOptimizer(lr_main)  # optimizer for the backbone network
        optimizer_sub = tf.train.AdamOptimizer(lr_sub)  # optimizer for the MPAE

        # loss of each decoder
        self.loss_ae = []
        # full loss of DAM-CNN
        self.loss_main = []
        train_step_sub = []
        train_step_main = []
        for d in range(0, self.datainfo_dict['n_dom']):
            # reconstructed loss of a decoder
            squ_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.x_exp[d] - self.AEs[d].decoder), axis=1))
            main_loss = reg_ae * squ_loss + self.loss

            self.loss_ae.append(squ_loss)
            self.loss_main.append(main_loss)
            train_step_sub.append(optimizer_sub.minimize(self.loss_ae[d], var_list=self.sub_var_list[d]))
            train_step_main.append(optimizer_main.minimize(self.loss_main[d], var_list=self.main_var_list))

        # begin training
        sess.run(tf.global_variables_initializer())
        self.iniflag = True

        # import image directory and label data
        file_dict = np.load(data_path)
        tr_file = file_dict['tr_file']
        tr_lab = file_dict['tr_lab']
        tt_file = file_dict['tt_file']
        tt_lab = file_dict['tt_lab']
        val_file = file_dict['val_file']
        val_lab = file_dict['val_lab']

        # sequence to store training sample index
        sample_seq = np.arange(len(tr_lab))
        shuffle_seq = sample_seq.copy()
        # buffers to store loss (reconstructed loss and classification loss)
        p_loss_ae = np.zeros(self.datainfo_dict['n_dom'])
        p_cross_entropy = np.zeros(self.datainfo_dict['n_dom'])

        e = 1
        # training on FER2013
        while e <= epoch:

            ep_st = time.time()

            # shuffle the sample index of all decoder
            dom_seq = util.shf_sub_dom_FER2013(tr_lab, self.datainfo_dict['n_dom'])

            # iterative steps
            iter_n = int(np.ceil(np.shape(shuffle_seq)[0] / bs))
            np.random.shuffle(shuffle_seq)
            # start training
            print('start training')
            for batch_trn in range(0, iter_n):
                if batch_trn != iter_n - 1:
                    batch_seq = shuffle_seq[batch_trn * bs : (batch_trn + 1) * bs]
                else:
                    batch_seq = shuffle_seq[batch_trn * bs :]

                # batch training images and labels
                batch_ori_name = tr_file[batch_seq]
                batch_y = tr_lab[batch_seq]
                batch_x_ori = util.load_FER2013_batch(batch_ori_name, batch_y)

                for d in range(0, self.datainfo_dict['n_dom']):
                    # batch index of each decoder
                    batch_dom_seq = dom_seq[d][batch_seq]
                    batch_dom_name = tr_file[batch_dom_seq]

                    # load batch images (expected reconstructed images) for each decoder
                    batch_x_exp = util.load_FER2013_batch(batch_dom_name, batch_y)

                    # salient expression features of expected reconstructed images
                    fea_exp = sess.run(self.masked_conv, feed_dict={self.x_in: batch_x_exp})
                    fea_exp = np.reshape(fea_exp, [fea_exp.shape[0], 7 * 7 * 512])

                    # update network parameters
                    p_loss, p_cross_entropy[d], p_loss_ae[d], _, _ = sess.run(
                        [self.loss, self.cross_entropy, self.loss_ae[d], train_step_main[d], train_step_sub[d]],
                        feed_dict={self.x_in: batch_x_ori, self.y_in: batch_y,
                                   self.x_exp[d]: fea_exp, self.keep_prob: 0.5, self.keep_prob_MPAE: 0.5})

                if batch_trn % 10 == 0:
                    # print the loss
                    print('the %d-th epoch cross_entropy:' % e)
                    print('%d / %d' % (batch_trn, iter_n))
                    print(p_cross_entropy)
                    print('loss of all decoders:')
                    print(p_loss_ae)
                    print('\n')

            # validation
            if e % 1 == 0:
                # validation set
                print('validation set:')
                self.val(sess, val_file, val_lab)
                # testing set
                print('testing set:')
                self.val(sess, tt_file, tt_lab, mode='test')

                # save model
                if self.tt_acc_mat[-1] > thr_acc:
                    thr_acc = self.tt_acc_mat[-1].copy()
                    self.save_par(sess)

            # end time of an epoch
            ep_et = time.time()
            # print time information
            print('an epoch last for %f seconds\n' % (ep_et - ep_st))

            e += 1

    def val(self, sess, val_name, val_lab, bs=32, mode='val'):
        '''
        validation on FER2013
        '''

        # sample numbers
        len_val = val_name.shape[0]

        # batch counting
        batch_valn = 0
        accuracy = 0
        val_ce = 0

        # randomly select samples
        val_randperm = (np.random.choice(np.arange(0, len_val), len_val, replace=False)).astype(int)

        while (len_val / bs) > batch_valn:
            if (batch_valn + 1) * bs > len_val:
                batch_ind = val_randperm[batch_valn * bs:]
            else:
                batch_ind = val_randperm[batch_valn * bs: (batch_valn + 1) * bs]

            # batch images and labels
            batch_name = val_name[batch_ind]
            batch_y = val_lab[batch_ind]
            batch_x = util.load_FER2013_batch(batch_name, batch_y, mode)

            val_dict = {self.x_in: batch_x, self.y_in: batch_y, self.keep_prob: 1}
            pred_temp, ce_temp = sess.run([self.result, self.cross_entropy], feed_dict=val_dict)
            accuracy_temp = util.acc_cal2(pred_temp, batch_y)
            accuracy = accuracy + accuracy_temp
            val_ce = val_ce + ce_temp
            batch_valn += 1

        acc_actual = accuracy / batch_valn
        ce_actual = val_ce / batch_valn

        if mode == 'val':
            self.val_acc_mat.append(acc_actual)
            print('the validation accuracy is : %f' % acc_actual)
            print('the validation cross entropy is %f \n' % ce_actual)
        else:
            self.tt_acc_mat.append(acc_actual)
            print('the testing accuracy is : %f' % acc_actual)
            print('the testing cross entropy is %f \n' % ce_actual)

    def save_par(self, sess):
        '''
        save the parameters of the trained model
        before saving the parameters, self.build() should be invoked
        :param sess: the session
        :return: None
        '''

        file_name = 'DAMCNN_par.npy'
        result_name = 'DAMCNN_result.npy'
        file_dir = os.path.join(self.datainfo_dict['db_name'], str(self.datainfo_dict['n_dom']))
        result_dir = os.path.join(self.datainfo_dict['db_name'] + '_res', str(self.datainfo_dict['n_dom']))
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # complete directory to save
        cpt_dir_par = os.path.join(file_dir, file_name)
        cpt_dir_result = os.path.join(result_dir, result_name)

        conv_depth = 5
        # dictionary to store network parameters
        data_dict = {}
        result_dict = {}
        # initialize variables
        if not self.iniflag:
            sess.run(tf.global_variables_initializer())

        # load and save parameters
        for i in range(1, 9):
            if i <= conv_depth:
                for conv_ind in range(1, 4):
                    # save convolution layer's parameters
                    layer_name = 'conv' + str(i) + '_' + str(conv_ind)
                    if layer_name not in self.par_dict.keys():
                        break
                    kernel, bias = self.get_var(layer_name)
                    if layer_name not in data_dict.keys():
                        data_dict[layer_name] = {}

                    d_kernel, d_bias = sess.run([kernel, bias])
                    data_dict[layer_name]['kernel'] = d_kernel
                    data_dict[layer_name]['bias'] = d_bias
            else:
                # save fc layers' parameter
                layer_name = 'fc' + str(i)
                kernel, bias = self.get_var(layer_name)
                if layer_name not in data_dict.keys():
                    data_dict[layer_name] = {}

                d_kernel, d_bias = sess.run([kernel, bias])
                data_dict[layer_name]['kernel'] = d_kernel
                data_dict[layer_name]['bias'] = d_bias

        # save the parameter of attention layer
        layer_name = 'att'
        kernel, bias = self.get_var(layer_name)
        if layer_name not in data_dict.keys():
            data_dict[layer_name] = {}

        d_kernel, d_bias = sess.run([kernel, bias])
        data_dict[layer_name]['kernel'] = d_kernel
        data_dict[layer_name]['bias'] = d_bias

        # save the parameter of decoders
        for i in range(0, self.datainfo_dict['n_dom']):
            layer_name = 'dec' + str(i)
            kernel, bias = self.get_var(layer_name)
            if layer_name not in data_dict.keys():
                data_dict[layer_name] = {}

            d_kernel, d_bias = sess.run([kernel, bias])
            data_dict[layer_name]['kernel'] = d_kernel
            data_dict[layer_name]['bias'] = d_bias

        # save accuracy matric
        result_dict['tt_acc'] = self.tt_acc_mat.copy()
        result_dict['val_acc'] = self.val_acc_mat.copy()

        self.model_par_dir = cpt_dir_par
        self.result_dir = cpt_dir_result

        print('now saving the %d-th epoch model' % (len(self.tt_acc_mat)))
        np.save(self.model_par_dir, data_dict)
        np.save(self.result_dir, result_dict)

    def get_var(self, layer_name):
        '''
        retrieve the desired variables (kernels and bias) in the network
        '''
        kernel_name = layer_name + '/kernel:'
        bias_name = layer_name + '/bias:'
        kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, kernel_name)
        bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, bias_name)
        return kernel[0], bias[0]

class AE(object):
    def __init__(self, in_data, kp, ae_n):
        '''
        a Decoder of MPAE
        :param in_data: input data
        :param kp: dropout keep probability
        :param ae_n: domain number of data
        '''

        self.keep_prob = kp
        self.var_list = []
        with tf.name_scope('dec' + str(ae_n)):
            self.decoder = self.decode(in_data, 'dec' + str(ae_n), 'sigmoid')

    def decode(self, in_data, name, activation='sigmoid'):
        # decode the input data
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='kernel',
                                     initializer=tf.truncated_normal([4096, 7*7*512], stddev=0.1))
            bias = tf.get_variable(name='bias', initializer=tf.zeros([7*7*512]))

        # append the variables of decoder to the parameter list
        self.var_list.append(kernel)
        self.var_list.append(bias)

        # decoding
        mul_add = tf.nn.bias_add(tf.matmul(tf.reshape(in_data, [-1, 4096]), kernel), bias)
        if  activation == 'relu':
            mul_add = tf.nn.relu(mul_add)
        elif activation == 'tanh':
            mul_add = tf.nn.tanh(mul_add)
        else:
            mul_add = tf.nn.sigmoid(mul_add)

        return tf.nn.dropout(mul_add, keep_prob=self.keep_prob)


