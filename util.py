import numpy as np
import os
import scipy.misc as sm


def load_FER2013_batch(batch_name, batch_lab, state='train'):
    '''
    load batch data from FER2013 dataset
    :param batch_name: image file names of this batch
    :param batch_lab:  image labels
    :param state: the sample set for this batch
    :return: a batch of images
    '''

    folder_dir = os.path.join('DB/FER2013/', state)
    lab_ind = np.argmax(batch_lab, 1)

    # buffer to store batch images
    img = []
    for i, cls in enumerate(lab_ind):
        img_dir = os.path.join(folder_dir, str(cls), batch_name[i])
        single_img = sm.imread(img_dir, mode='RGB')
        single_img = sm.imresize(single_img, [224, 224]) / 255
        img.append(single_img)

    return img

# shuffle the sample index of each decoder in MPAE for FER2013 dataset
def shf_sub_dom_FER2013(lab, dom_n):
    '''
    :param lab: sample labels
    :param dom_n: number of decoding paths
    :return: a list of shuffled sample index
    '''

    cls_n = np.shape(lab)[1]
    dom_seq = []
    for i in range(cls_n):
        # sample index of the i-th class
        ind_i = np.where(lab[:, i] == 1)[0]
        temp_i = []
        for d in range(dom_n):
            if d == 0:
                np.random.shuffle(ind_i)
                temp_i = ind_i.copy()
            else:
                np.random.shuffle(ind_i)
                temp_i = np.vstack((temp_i, ind_i.copy()))

        if i == 0:
            dom_seq = temp_i
        else:
            dom_seq = np.hstack((dom_seq, temp_i))

    return dom_seq


def acc_cal2(pred, test_y):
    pred_seq = np.argmax(pred, 1)
    gt_seq = np.argmax(test_y, 1)
    acc = np.mean((pred_seq == gt_seq).astype(float))
    return acc


