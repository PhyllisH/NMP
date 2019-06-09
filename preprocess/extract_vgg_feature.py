'''
	Extract features by pretrained VGG checkpoints
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import numpy as np 
from ass_fun import *
from vgg import VTranse_Vgg
import ipdb
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='vrd',
                    help='dataset: vrd or vg')
parser.add_argument('--data_type', type=str, default='pred',
                    help='data_type: pred or rela')
parser.add_argument('--ori_vgg', action='store_true', default=False,
                    help='original vgg')
parser.add_argument('--random_vgg', action='store_true', default=False,
                    help='random initialize vgg')

args = parser.parse_args()
data_type = args.data_type
dataset = args.dataset
use_ori_vgg = args.ori_vgg
use_random_vgg = args.random_vgg

feat_save_path = '/DATA5_DB8/data/yhu/VTransE'

print(args)

if dataset == 'vrd' and data_type == 'pred':
	# ---------- vrd pred dataset ---------------#
	if use_ori_vgg:
		save_path = os.path.join(feat_save_path, 'ori_vrd_vgg_feats')
	elif use_random_vgg:
		save_path = os.path.join(feat_save_path, 'random_vrd_vgg_feats')
	else:
		save_path = os.path.join(feat_save_path, 'vrd_vgg_feats')
	roidb_path = '../data/vrd_roidb.npz'
	res_path = '../data/pretrained/vrd_vgg_pretrained.ckpt'
	N_each_batch = 30
	is_rela = False
elif dataset == 'vrd' and data_type == 'rela':
	# ---------- vrd rela dataset ----------#
	if use_ori_vgg:
		save_path = os.path.join(feat_save_path, 'ori_vrd_rela_vgg_feats')
	elif use_random_vgg:
		save_path = os.path.join(feat_save_path, 'random_vrd_rela_vgg_feats')
	else:
		save_path = os.path.join(feat_save_path, 'vrd_rela_vgg_feats')
	roidb_path = '../data/vrd_rela_roidb.npz'
	res_path = '../data/pretrained/vrd_vgg_pretrained.ckpt'
	N_each_batch = 50
	is_rela = True
elif dataset == 'vg' and data_type == 'pred':
	# ----------- vg dataset ---------------#
	if use_ori_vgg:
		save_path = os.path.join(feat_save_path, 'ori_vg_vgg_feats')
	else:
		save_path = os.path.join(feat_save_path, 'vg_vgg_feats')
	roidb_path = '../data/vg_roidb.npz'
	res_path = '../data/pretrained/vg_vgg_pretrained.ckpt'
	N_each_batch = 30
	is_rela = False
elif dataset == 'vg' and data_type == 'rela':
	# ----------- vg rela dataset ---------------#
	save_path = os.path.join(feat_save_path, 'vg_rela_vgg_feats')
	roidb_path = '../data/vg_rela_roidb.npz'
	res_path = '../data/pretrained/vg_vgg_pretrained.ckpt'
	N_each_batch = 128
	is_rela = True

check_path_exists(save_path)

# ------ read roidb file ---------#
roidb_read = read_roidb(roidb_path)
train_roidb = roidb_read['train_roidb']
test_roidb = roidb_read['test_roidb']
N_train = len(train_roidb)
N_test = len(test_roidb)
pbar = tqdm(total=N_train+N_test)
N_show = 100

# ------ Create Graph ------------#
vnet = VTranse_Vgg()
graph_name = vnet.create_graph
train_func = vnet.extract_pred_fc
test_func = vnet.extract_pred_fc
graph_name(N_each_batch, save_path)

total_var = tf.trainable_variables()
restore_var = [var for var in total_var if 'vgg_16' in var.name]
for var in restore_var:
	print(var)
saver_res = tf.train.Saver(var_list = restore_var)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)

	if use_ori_vgg:
		# ------ restore from original vgg ---------#
		restore_from_npy(sess, restore_var)
	elif use_random_vgg:
		pass
	else:
		# ------ restore from fine-tuned vgg -------#
		saver_res.restore(sess, res_path)
	# ipdb.set_trace()
	t = 0.0
	vnet.save_path = save_path + '/train'
	check_path_exists(vnet.save_path)
	for roidb_id in range(N_train):
		roidb_use = train_roidb[roidb_id]
		if len(roidb_use['rela_gt']) == 0:
			continue
		if os.path.exists(os.path.join(vnet.save_path, 'ob_fc7', os.path.basename(roidb_use['image'])+'.npy')):
			pass
		else:
			train_func(sess, roidb_use, is_rela)
		t = t + 1.0
		if t % N_show == 0:
			print("t: {0}".format(t))
		pbar.update(1)
	vnet.save_path = save_path + '/test'
	check_path_exists(vnet.save_path)
	for roidb_id in range(N_test):
		roidb_use = test_roidb[roidb_id]
		if len(roidb_use['rela_gt']) == 0:
			continue
		if os.path.exists(os.path.join(vnet.save_path, 'ob_fc7', os.path.basename(roidb_use['image'])+'.npy')):
			pass
		else:
			test_func(sess, roidb_use, is_rela)
		t = t + 1.0
		if t % N_show == 0:
			print("t: {0}".format(t))
		pbar.update(1)
pbar.close()
