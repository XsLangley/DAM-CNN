'''
training DAM-CNN on the FER2013 dataset
'''

import tensorflow as tf
import numpy as np
import os
import DAMCNN_end2end


# gpu configure
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True

# database name
db_name = 'FER2013'

# the directory of trained vgg-face model
vggdir = 'vgg_origin_dict.npy'

# number of decoding path
n_dom = 2

#region start training

# expression classes number
cls_num = 7

# dictionary to store data information
datainfo_dict = {'db_name': db_name, 'n_dom': n_dom, 'cls_num': cls_num}

# directory of sample name and labels
imgdir = 'DB/FER2013/file_lab_fer2013.npz'

# path of saved model
par_path = os.path.join(datainfo_dict['db_name'], str(datainfo_dict['n_dom']), 'DAMCNN_par.npy')

if not (os.path.isfile(par_path)):
    DAMCNN = DAMCNN_end2end.DAMCNN(infor_dict=datainfo_dict, vgg_path=vggdir)

    DAMCNN_garaph = tf.Graph()
    with tf.Session(graph=DAMCNN_garaph, config=gpu_config) as sess:
        print('\nstart training')

        DAMCNN.build()
        DAMCNN.train(sess, imgdir, epoch=10, lr_main=0.5*1e-5, lr_sub=1e-4, reg_2=5*1e-4, bs=32)
        # print accuracy vector
        print('the accuracy vector is:')
        print(DAMCNN.tt_acc_mat)
        print('the best accuracy of DAM-CNN is: %f' % np.max(DAMCNN.tt_acc_mat))
        print('the index of the best accuracy is:')
        print(np.where(DAMCNN.tt_acc_mat == np.max(DAMCNN.tt_acc_mat)))


#endregion

print('end')
