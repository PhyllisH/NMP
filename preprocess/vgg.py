from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import numpy as np
from ass_fun import *
import ipdb
import os

class VTranse_Vgg(object):
	def __init__(self):
		self.predictions = {}
		self.losses = {}
		self.layers = {}
		self.feat_stride = [16, ]
		self.scope = 'vgg_16'

	def create_graph(self, batch_size, save_path):
		# extract subject and object feature
		# rela: test and pred: train & test
		self.image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
		self.sbox = tf.placeholder(tf.float32, shape=[batch_size, 4])		#[x1, y1, x2, y2]
		self.obox = tf.placeholder(tf.float32, shape=[batch_size, 4])		#[x1, y1, x2, y2]
		self.sub_sp_info = tf.placeholder(tf.float32, shape=[batch_size, 4])	# ???
		self.ob_sp_info = tf.placeholder(tf.float32, shape=[batch_size, 4])
		# self.rela_label = tf.placeholder(tf.int32, shape=[batch_size,])
		self.keep_prob = tf.placeholder(tf.float32)
		self.save_path = save_path
		self.batch_size = batch_size
		self.build_dete_network()

	def build_dete_network(self, is_training=True):
		# get the region conv and fc features 
		# the classfication probabilities and ids
		net_conv = self.image_to_head(is_training)
		net_pool5 = self.crop_bottom_layer(net_conv, "pool5")				# [n, 7, 7]
		sub_pool5 = self.crop_pool_layer(net_conv, self.sbox, "sub_pool5")	# [n, 7, 7]
		ob_pool5 = self.crop_pool_layer(net_conv, self.obox, "ob_pool5")	# [n, 7, 7]
		net_fc7 = self.head_to_tail(net_pool5, is_training, reuse = False)	# [n, 4096]
		sub_fc7 = self.head_to_tail(sub_pool5, is_training, reuse = True)	# [n, 4096]
		ob_fc7 = self.head_to_tail(ob_pool5, is_training, reuse = True)		# [n, 4096]

		# --------new added----------------#
		pred_pool5 = self.crop_union_pool_layer(net_conv, self.sbox, self.obox, "pred_pool5")	# [n, 7, 7]
		# pred_fc7 = self.head_to_tail(pred_pool5, is_training, reuse = True)
		pred_fc7 = self.head_to_mean_tail(pred_pool5, is_training, reuse = True)

		self.layers['sub_pool5'] = sub_pool5
		self.layers['ob_pool5'] = ob_pool5
		self.layers['sub_fc7'] = sub_fc7
		self.layers['ob_fc7'] = ob_fc7
		self.layers['pool5'] = net_pool5
		self.layers['fc7'] = net_fc7

		# --------new added----------------#
		self.layers['pred_pool5'] = pred_pool5
		self.layers['pred_fc7'] = pred_fc7

	def image_to_head(self, is_training, reuse=False):
		with tf.variable_scope(self.scope, self.scope, reuse=reuse):
			net = slim.repeat(self.image, 2, slim.conv2d, 64, [3, 3], 
				trainable=is_training, scope='conv1')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
			net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
				trainable=is_training, scope='conv2')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
			net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
				trainable=is_training, scope='conv3')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
				trainable=is_training, scope='conv4')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
			net_conv = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
				trainable=is_training, scope='conv5')

			self.layers['head'] = net_conv
			return net_conv

	def head_to_tail(self, pool5, is_training, reuse=False):
		with tf.variable_scope(self.scope, self.scope, reuse=reuse):
			pool5_flat = slim.flatten(pool5, scope='flatten')	#[n, 49]
			fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
			fc6 = slim.dropout(fc6, keep_prob=self.keep_prob, is_training=True, 
					scope='dropout6')
			fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
			fc7 = slim.dropout(fc7, keep_prob=self.keep_prob, is_training=True, 
					scope='dropout7')

			return fc7

	def head_to_mean_tail(self, pool5, is_training, reuse=False):
		mean_fc7 = tf.reduce_mean(tf.reduce_mean(pool5, axis=2), axis=1)
		return mean_fc7

	def crop_pool_layer(self, bottom, rois, name):
		"""
		Notice that the input rois is a N*4 matrix, and the coordinates of x,y should be original x,y times im_scale. 
		"""
		with tf.variable_scope(name) as scope:
			n=tf.to_int32(rois.shape[0])
			batch_ids = tf.zeros([n,],dtype=tf.int32)
			# Get the normalized coordinates of bboxes
			bottom_shape = tf.shape(bottom)
			height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.feat_stride[0])
			width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.feat_stride[0])
			# separate the (x1, y1, x2, y2) of the bounding boxes' coordinates
			x1 = tf.slice(rois, [0, 0], [-1, 1], name="x1") / width
			y1 = tf.slice(rois, [0, 1], [-1, 1], name="y1") / height
			x2 = tf.slice(rois, [0, 2], [-1, 1], name="x2") / width
			y2 = tf.slice(rois, [0, 3], [-1, 1], name="y2") / height
			# Won't be back-propagated to rois anyway, but to save time
			bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))	#[n, 4]
			crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [7*2, 7*2], method='bilinear',
											 name="crops")
			pooling = max_pool(crops, 2, 2, 2, 2, name="max_pooling")
		return pooling

	def crop_union_pool_layer(self, bottom, rois_s, rois_o, name):
		"""
		Notice that the input rois is a N*4 matrix, and the coordinates of x,y should be original x,y times im_scale. 
		"""
		with tf.variable_scope(name) as scope:
			n=tf.to_int32(rois_s.shape[0])
			batch_ids = tf.zeros([n,],dtype=tf.int32)
			# Get the normalized coordinates of bboxes
			bottom_shape = tf.shape(bottom)
			height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.feat_stride[0])
			width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.feat_stride[0])
			# separate the (x1, y1, x2, y2) of the bounding boxes' coordinates
			x1_s = tf.slice(rois_s, [0, 0], [-1, 1], name="x1_s")
			y1_s = tf.slice(rois_s, [0, 1], [-1, 1], name="y1_s") 
			x2_s = tf.slice(rois_s, [0, 2], [-1, 1], name="x2_s")
			y2_s = tf.slice(rois_s, [0, 3], [-1, 1], name="y2_s") 

			x1_o = tf.slice(rois_o, [0, 0], [-1, 1], name="x1_o")
			y1_o = tf.slice(rois_o, [0, 1], [-1, 1], name="y1_o") 
			x2_o = tf.slice(rois_o, [0, 2], [-1, 1], name="x2_o")
			y2_o = tf.slice(rois_o, [0, 3], [-1, 1], name="y2_o")

			x1 = tf.minimum(x1_s, x1_o, name="x1") / width
			y1 = tf.minimum(y1_s, y1_o, name="y1") / height
			x2 = tf.maximum(x2_s, x2_o, name="x2") / width
			y2 = tf.maximum(y2_s, y2_o, name="y2") / height

			# Won't be back-propagated to rois anyway, but to save time
			bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))	#[n, 4]
			crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [7*2, 7*2], method='bilinear',
											 name="crops")
			pooling = max_pool(crops, 2, 2, 2, 2, name="max_pooling")
		return pooling

	def crop_bottom_layer(self, bottom, name):
		"""
		Notice that the input rois is a N*4 matrix, and the coordinates of x,y should be original x,y times im_scale. 
		"""
		with tf.variable_scope(name) as scope:
			# Get the normalized coordinates of bboxes
			resized = tf.image.resize_images(bottom, [7*2, 7*2])
			pooling = max_pool(resized, 2, 2, 2, 2, name="max_pooling")
		return pooling

	def extract_pred_fc(self, sess, roidb_use, is_rela=False):
		im, im_scale = im_preprocess(roidb_use['image'])
		if is_rela:
			batch_num = len(roidb_use['index_rela'])/self.batch_size
		else:
			batch_num = len(roidb_use['index_pred'])/self.batch_size
		
		layers = []
		keys = ['pred_pool5', 'pred_fc7', 'pool5', 'fc7', 'sub_fc7', 'ob_fc7']
		for k in keys:
			check_path_exists(os.path.join(self.save_path, k))
		for batch_id in range(np.int32(batch_num)):
			if is_rela:
				blob = get_blob_rela(roidb_use, im_scale, self.batch_size, batch_id)
			else:
				blob = get_blob_pred(roidb_use, im_scale, self.batch_size, batch_id)
			
			feed_dict = {self.image: im, self.sbox: blob['sub_box'], self.obox: blob['obj_box'],
						 self.keep_prob: 1}
			layer = sess.run(self.layers, feed_dict = feed_dict)
			layer_feat = map(lambda x: layer[x], keys)
			layers.append(layer_feat)
			
		pred_pool5 = []
		pred_fc7 = []
		pool5 = []
		fc7 = []
		sub_fc7 = []
		ob_fc7 = []
		for i in range(len(layers)):
			pred_pool5.append(layers[i][0])
			pred_fc7.append(layers[i][1])
			pool5.append(layers[i][2])
			fc7.append(layers[i][3])
			sub_fc7.append(layers[i][4])
			ob_fc7.append(layers[i][5])
		pred_pool5 = np.concatenate(pred_pool5, 0)
		pred_fc7 = np.concatenate(pred_fc7, 0)
		pool5 = np.concatenate(pool5, 0)
		fc7 = np.concatenate(fc7, 0)
		sub_fc7 = np.concatenate(sub_fc7, 0)
		ob_fc7 = np.concatenate(ob_fc7, 0)

		if is_rela:
			n_total = len(roidb_use['rela_dete'])
		else:
			n_total = len(roidb_use['rela_gt'])

		pred_pool5_full_save_path = os.path.join(self.save_path, 'pred_pool5', os.path.basename(roidb_use['image']))
		pred_fc7_full_save_path = os.path.join(self.save_path, 'pred_fc7', os.path.basename(roidb_use['image']))
		pool5_full_save_path = os.path.join(self.save_path, 'pool5', os.path.basename(roidb_use['image']))
		fc7_full_save_path = os.path.join(self.save_path, 'fc7', os.path.basename(roidb_use['image']))
		sub_fc7_full_save_path = os.path.join(self.save_path, 'sub_fc7', os.path.basename(roidb_use['image']))
		ob_fc7_full_save_path = os.path.join(self.save_path, 'ob_fc7', os.path.basename(roidb_use['image']))
		np.save(pred_pool5_full_save_path, pred_pool5[:n_total])
		np.save(pred_fc7_full_save_path, pred_fc7[:n_total])
		np.save(pool5_full_save_path, pool5[:n_total])
		np.save(fc7_full_save_path, fc7[:n_total])
		np.save(sub_fc7_full_save_path, sub_fc7[:n_total])
		np.save(ob_fc7_full_save_path, ob_fc7[:n_total])
		print("{0} processed!".format(roidb_use['image']))
		return

def max_pool(x, h, w, s_y, s_x, name, padding='SAME'):
	return tf.nn.max_pool(x, ksize=[1,h,w,1], strides=[1, s_x, s_y, 1], padding=padding, name=name)
