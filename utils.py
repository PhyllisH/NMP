from __future__ import print_function
import numpy as np
import os
import ipdb
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# def read_roidb(roidb_path):
# 	'''python2'''
# 	roidb_file = np.load(roidb_path)
# 	key = roidb_file.keys()[0]
# 	roidb_temp = roidb_file[key]
# 	roidb = roidb_temp[()]
# 	return roidb
def compute_acc(output, target, ignored_index):
    '''
        output : [N, N_cls]
        target : [N,]; GT category
        ignored_index: int; the category that does not count
    '''
    pred = output.data.max(1, keepdim=True)[1]
    count_mask = (target < ignored_index)
    correct = (pred.eq(target.data.view_as(pred)) * count_mask.view(-1,1).data).cpu().sum()
    count = count_mask.data.cpu().sum()
    if count < 0.1:
        acc = 0
    else:
        acc = correct.float()/count.float()
    return acc.item()

def compute_iou_each(box1, box2):
    '''
    function: calculate the iou based on the box ordinates
    box1: [x_min, y_min, x_max, y_max]
    '''
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    if xB<xA or yB<yA:
        IoU = 0
    else:
        area_I = (xB - xA + 1) * (yB - yA + 1)
        area1 = (box1[2] - box1[0] + 1)*(box1[3] - box1[1] + 1)
        area2 = (box2[2] - box2[0] + 1)*(box2[3] - box2[1] + 1)
        IoU = area_I/float(area1 + area2 - area_I)
    return IoU

def compute_distance(box1, box2):
    cx1 = (box1[0] + box1[2])/2.0
    cy1 = (box1[1] + box1[3])/2.0
    cx2 = (box2[0] + box2[2])/2.0
    cy2 = (box2[1] + box2[3])/2.0

    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])

    I = (cx1 - cx2)**2 + (cy1 - cy2)**2
    U = (x_min - x_max)**2 + (y_min - y_max)**2
    dis = np.sqrt(I/float(U))
    return dis

def get_box_feats(sub_box, obj_box):
    '''
    box: [x_min, y_min, x_max, y_max]
    '''
    def _center(box):
        x_c = (box[0] + box[2])/2.0
        y_c = (box[1] + box[3])/2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        return np.array([x_c, y_c, w, h])
    def _union(box1, box2):
        x_min = min(box1[0], box2[0])
        y_min = min(box1[1], box2[1])
        x_max = max(box1[2], box2[2])
        y_max = max(box1[3], box2[3])
        return np.array([x_min, y_min, x_max, y_max])
    def _six(c_sub_box, c_obj_box):
        t_x_so = (c_sub_box[0] - c_obj_box[0])/float(c_sub_box[2])
        t_y_so = (c_sub_box[1] - c_obj_box[1])/float(c_sub_box[3])
        t_w_so = np.log(c_sub_box[2]/float(c_obj_box[2]))
        t_h_so = np.log(c_sub_box[3]/float(c_obj_box[3]))
        t_x_os = (c_obj_box[0] - c_sub_box[0])/float(c_obj_box[2])
        t_y_os = (c_obj_box[1] - c_sub_box[1])/float(c_obj_box[3])
        return np.array([t_x_so, t_y_so, t_w_so, t_h_so, t_x_os, t_y_os])

    p_box = _union(sub_box, obj_box)
    c_sub_box = _center(sub_box)
    c_obj_box = _center(obj_box)
    c_p_box = _center(p_box)
    six_so = _six(c_sub_box, c_obj_box)
    six_sp = _six(c_sub_box, c_p_box)
    six_op = _six(c_obj_box, c_p_box)

    iou = compute_iou_each(sub_box, obj_box)
    dis = compute_distance(sub_box, obj_box)
    iou_dis = np.array([iou, dis])

    output = np.concatenate([six_so, six_sp, six_op, iou_dis],0)
    return output

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def read_roidb(roidb_path):
	''' python3 '''
	roidb_file = np.load(roidb_path, encoding='latin1')
	key = list(roidb_file.keys())[0]
	roidb_temp = roidb_file[key]
	roidb = roidb_temp[()]
	return roidb

def box_id(ori_box, uni_box):
	'''
	input:
		ori_box: the sub or obj box ordinates
		uni_box: the unique box ordinates
	output:
		the idx of the ori_box based on the unique box
	'''
	idx = []
	for i in range(len(ori_box)):
		for j in range(len(uni_box)):
			if np.array_equal(ori_box[i], uni_box[j]):
				idx.append(j)
	return idx

def compute_iou_each(box1, box2):
	'''
	function: calculate the iou based on the box ordinates
	box1: [x_min, y_min, x_max, y_max]
	'''
	xA = max(box1[0], box2[0])
	yA = max(box1[1], box2[1])
	xB = min(box1[2], box2[2])
	yB = min(box1[3], box2[3])

	if xB<xA or yB<yA:
		IoU = 0
	else:
		area_I = (xB - xA + 1) * (yB - yA + 1)
		area1 = (box1[2] - box1[0] + 1)*(box1[3] - box1[1] + 1)
		area2 = (box2[2] - box2[0] + 1)*(box2[3] - box2[1] + 1)
		IoU = area_I/float(area1 + area2 - area_I)
	return IoU

def get_item(arr, idx, idy):
	out = np.zeros(len(idx))
	for i in range(len(idx)):
		out[i] = arr[idx[i], idy[i]]
	return out

def encode_onehot(labels):
	classes = set(labels)
	classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
						enumerate(classes)}
	labels_onehot = np.array(list(map(classes_dict.get, labels)),
						dtype=np.int32)
	return labels_onehot

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]

class FocalLoss(nn.Module):
    def __init__(self, num_classes=20, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)  # [N,D]
        t = t[:,:self.num_classes]  # exclude background
        t = Variable(t).cuda()  # [N,D-1]

        x = x[:,:self.num_classes]
        p = F.softmax(x, dim=-1)
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = self.alpha*t + (1-self.alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(self.gamma)
        return F.binary_cross_entropy_with_logits(p.log(), t, w, reduction='none').sum(-1)