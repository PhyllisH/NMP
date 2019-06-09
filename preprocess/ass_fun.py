import numpy as np 
import cv2 
import os
import json
import ipdb

def restore_from_npy(sess, restore_var):
	vgg_npy = np.load('../data/pretrained/VGG_imagenet.npy')
	vgg_npy = vgg_npy[()]
	keys_1 = ['conv1_1', 'conv1_1', 'conv1_2', 'conv1_2', \
			  'conv2_1', 'conv2_1', 'conv2_2', 'conv2_2', \
			  'conv3_1', 'conv3_1', 'conv3_2', 'conv3_2', 'conv3_3', 'conv3_3', \
			  'conv4_1', 'conv4_1', 'conv4_2', 'conv4_2', 'conv4_3', 'conv4_3', \
			  'conv5_1', 'conv5_1', 'conv5_2', 'conv5_2', 'conv5_3', 'conv5_3', \
			  'fc6', 'fc6', 'fc7', 'fc7']
	keys_2 = ['weights', 'biases', 'weights', 'biases', \
			  'weights', 'biases', 'weights', 'biases', \
			  'weights', 'biases', 'weights', 'biases', 'weights', 'biases', \
			  'weights', 'biases', 'weights', 'biases', 'weights', 'biases', \
			  'weights', 'biases', 'weights', 'biases', 'weights', 'biases', \
			  'weights', 'biases', 'weights', 'biases']
	for ind, var in enumerate(restore_var):
		sess.run(var.assign(vgg_npy[keys_1[ind]][keys_2[ind]]))
	return

def read_roidb(roidb_path):
	'''python2'''
	roidb_file = np.load(roidb_path)
	key = roidb_file.keys()[0]
	roidb_temp = roidb_file[key]
	roidb = roidb_temp[()]
	return roidb

def generate_batch(N_total, N_each):
	"""
	This file is used to generate index of the training batch.
	
	Arg:
		N_total: 
		N_each: 
	out_put: 
		index_box: the corresponding index
		if the total number can divide the batch_num, just split them
		else enlarge the index set to be the minimum miltiple
		and randomly choose from the total set as the padding indexes
	"""
	num_batch = np.int32(N_total/N_each)
	if N_total%N_each == 0:
		index_box = range(N_total)
	else:
		index_box = np.empty(shape=[N_each*(num_batch+1)],dtype=np.int32)
		index_box[0:N_total] = range(N_total)
		N_rest = N_each*(num_batch+1) - N_total
		index_box[N_total:] = np.random.randint(0,N_total,N_rest)
	return index_box

def check_path_exists(full_log_dir):
	if os.path.exists(full_log_dir):
		pass
	else:
		os.mkdir(full_log_dir)

def generate_rela_info(au_box, index, N_each_pair):
	s_id = np.int32(index[0])
	o_id = np.int32(index[1])
	sbox = au_box[s_id]
	obox = au_box[o_id]
	N_s = len(sbox)
	N_o = len(obox)
	sa = np.random.randint(0, N_s, [N_each_pair,])	# randomly extract N_each_pair(5 in the config) of the detected boxes
	oa = np.random.randint(0, N_o, [N_each_pair,])	# whose iou larger than the threhold
	sbox_use = sbox[sa]
	obox_use = obox[oa]
	return sbox_use, obox_use

def box_id(ori_box, uni_box):
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

def iou_dis(iou_thre=0.5, dis_thre=0.45):
	roidb = read_roidb('./data/vrd_rela_graph_roidb.npz')
	train = roidb['train']
	test = roidb['test']
	new_roidb_test = []
	for i in range(len(test)):
		new_roidb_use = copy.deepcopy(test[i])
		roidb_use = test[i]
		keep_index = []
		for j in range(len(roidb_use['sub_box_dete'])):
			sub_box = roidb_use['sub_box_dete'][j]
			obj_box = roidb_use['obj_box_dete'][j]
			iou = compute_iou_each(sub_box, obj_box)
			dis = compute_distance(sub_box, obj_box)
			if (iou>iou_thre) or (dis<dis_thre):
				keep_index.append(j)
			new_roidb_use['sub_box_dete'] = roidb_use['sub_box_dete'][keep_index]
			new_roidb_use['obj_box_dete'] = roidb_use['obj_box_dete'][keep_index]
			new_roidb_use['sub_dete'] = roidb_use['sub_dete'][keep_index]
			new_roidb_use['obj_dete'] = roidb_use['obj_dete'][keep_index]
			new_roidb_use['rela_dete'] = roidb_use['rela_dete'][keep_index]
			new_roidb_use['sub_score'] = roidb_use['sub_score'][keep_index]
			new_roidb_use['obj_score'] = roidb_use['obj_score'][keep_index]
		#	print(j, len(keep_index), len(roidb_use['sub_box_dete']))
		new_roidb_test.append(new_roidb_use)
	# save the object pairs which meet the <iou-dis> constrain
	new_roidb = {}
	new_roidb['train'] = roidb['train']
	new_roidb['test'] = new_roidb_test
	np.savez('./data/graph_roidb_iou_dis_{}_{}.npz'.format(iou_thre*10, dis_thre*10), new_roidb)

def compute_iou(box, proposal):
	"""
	compute the IoU between box with proposal
	Arg:
		box: [x1,y1,x2,y2]
		proposal: N*4 matrix, each line is [p_x1,p_y1,p_x2,p_y2]
	output:
		IoU: N*1 matrix, every IoU[i] means the IoU between
			 box with proposal[i,:]
	"""
	len_proposal = np.shape(proposal)[0]
	IoU = np.empty([len_proposal,1])
	for i in range(len_proposal):
		xA = max(box[0], proposal[i,0])
		yA = max(box[1], proposal[i,1])
		xB = min(box[2], proposal[i,2])
		yB = min(box[3], proposal[i,3])

		if xB<xA or yB<yA:
			IoU[i,0]=0
		else:
			area_I = (xB - xA + 1) * (yB - yA + 1)
			area1 = (box[2] - box[0] + 1)*(box[3] - box[1] + 1)
			area2 = (proposal[i,2] - proposal[i,0] + 1)*(proposal[i,3] - proposal[i,1] + 1)
			IoU[i,0] = area_I/float(area1 + area2 - area_I)
	return IoU


def generate_au_box(unique_boxes, detected_box, iou_l):
	# extract the detected_box whose iou is larger than anyone in the unique ground truth boxes
	# return [num(unique_boxeds) * [box_use 		<== multi detected boxes 
	#		  						box_temp]]		<== the ground truth box
	N_unique = len(unique_boxes)
	au_box = []
	for i in range(N_unique):
		box_temp = unique_boxes[i]
		iou = compute_iou(box_temp, detected_box)
		index_temp = np.where(iou > iou_l)[0]
		box_use = detected_box[index_temp]
		box_use = np.vstack( (box_use, box_temp ) )
		au_box.append(box_use)
	return au_box

def im_preprocess(image_path):
	image = cv2.imread(image_path)
	im_orig = image.astype(np.float32, copy=True)
	im_orig -= np.array([[[102.9801, 115.9465, 122.7717]]])

	im_shape = im_orig.shape
	im_size_min = np.min(im_shape[0:2])
	im_size_max = np.max(im_shape[0:2])

	target_size = 600
	max_size = 1000
	im_scale = float(target_size) / float(im_size_min)

	if np.round(im_scale * im_size_max) > max_size:
		im_scale = float(max_size) / float(im_size_max)
	# ipdb.set_trace()
	im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, 
		interpolation=cv2.INTER_LINEAR)
	im_shape_new = np.shape(im)
	im_use = np.zeros([1,im_shape_new[0], im_shape_new[1], im_shape_new[2]])
	im_use[0,:,:,:] = im
	return im_use, im_scale


def get_blob_pred(roidb_use, im_scale, N_each_batch, batch_id):
	blob = {}
	sub_box = roidb_use['sub_box_gt']*im_scale
	obj_box = roidb_use['obj_box_gt']*im_scale
	rela = np.int32(roidb_use['rela_gt'])
	index = roidb_use['index_pred']
	# spatial = roidb_use['spatial_gmm_vec']

	index_use = index[batch_id*N_each_batch: (batch_id+1)*N_each_batch]
	sub_box_use = sub_box[index_use,:]
	obj_box_use = obj_box[index_use,:]
	rela_use = rela[index_use]
	# spatial_use = spatial[index_use, :]

	blob['sub_box'] = sub_box_use
	blob['obj_box'] = obj_box_use
	blob['rela'] = rela_use
	# blob['spatial'] = spatial_use
	blob['image'] = roidb_use['image']
	return blob

def get_blob_rela(roidb_use, im_scale, N_each_batch, batch_id):
	blob = {}
	sub_box = roidb_use['sub_box_dete']*im_scale
	obj_box = roidb_use['obj_box_dete']*im_scale
	rela = np.int32(roidb_use['rela_dete'])
	index = roidb_use['index_rela']
	# spatial = roidb_use['spatial_gmm_vec']

	index_use = index[batch_id*N_each_batch: (batch_id+1)*N_each_batch]
	sub_box_use = sub_box[index_use,:]
	obj_box_use = obj_box[index_use,:]
	rela_use = rela[index_use]
	# spatial_use = spatial[index_use, :]

	blob['sub_box'] = sub_box_use
	blob['obj_box'] = obj_box_use
	blob['rela'] = rela_use
	# blob['spatial'] = spatial_use
	return blob

def count_prior():
	roidb = read_roidb('/DATA5_DB8/data/yhu/NRI/dsr_data/dsr_roidb.npz')
	train = roidb['train_roidb']
	prior = np.zeros([100, 100, 70])
	for i in range(len(train)):
		roidb_use = train[i]
		for j in range(len(roidb_use['rela_gt'])):
			sub_cls = int(roidb_use['sub_gt'][j])
			obj_cls = int(roidb_use['obj_gt'][j])
			rela_cls = int(roidb_use['rela_gt'][j])
			prior[sub_cls, obj_cls, rela_cls] += 1
	np.save('/DATA5_DB8/data/yhu/NRI/dsr_data/dsr_prior_count.npy', prior)
	prior_count = np.sum(prior, -1)
	prior_prob = prior_count/np.sum(prior_count)
	np.save('/DATA5_DB8/data/yhu/NRI/dsr_data/dsr_prior_prob.npy', prior_prob)
	return