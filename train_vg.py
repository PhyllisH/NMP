from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from modules import *
from eval_metrics import *
from utils import *
from DataLoader import *

import ipdb
from tqdm import tqdm
# from visualize import Visualizer

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Number of samples per batch.')
parser.add_argument('--eval-batch-size', type=int, default=32,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--num-atoms', type=int, default=110,
                    help='Number of atoms in simulation.')
parser.add_argument('--rela-num-atoms', type=int, default=63,
                    help='Number of atoms in simulation.')
parser.add_argument('--num-edges', type=int, default=490,
                    help='Number of atoms in simulation.')
parser.add_argument('--encoder', type=str, default='simple',
                    help='Type of path encoder model(simple or nmp).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='How many batches to wait before logging.')
parser.add_argument('--edge-types', type=int, default=101,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=4396,
                    help='The number of dimensions. 320/4396')
parser.add_argument('--save-folder', type=str, default='./checkpoints/vg',
                    help='Where to save the trained model.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model.')
parser.add_argument('--lr-decay', type=int, default=5,
                    help='After how epochs to decay LR by a factor of gamma')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor')
parser.add_argument('--weight', type=float, default=0,
                    help='Use motion capture data loader.')
parser.add_argument('--mode', type=str, default='whole',
                    help='Use motion capture data loader.')
parser.add_argument('--restore', action='store_true', default=False,
                    help='Restore the trained model from the load-folder.')
parser.add_argument('--shuffle', action='store_true', default=False,
                    help='Shuffle the data in the dataloader.')
parser.add_argument('--feat-mode', type=str, default='full',
                    help='feature mode: full, vis, or sem')
parser.add_argument('--n-iter', type=int, default=3,
                    help='How many times of the node edge transfer information.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Ranking loss')
parser.add_argument('--tail', type=str, default='base',
                    help='special name')
parser.add_argument('--ori-vgg', action='store_true', default=False,
                    help='original vgg')
parser.add_argument('--use-loc', action='store_true', default=False,
                    help='use location coordinates')
parser.add_argument('--use-cls', action='store_true', default=False,
                    help='add a classification layer and use the confidence score as feature')
parser.add_argument('--node-types', type=int, default=201,
                    help='The number of node types to infer.')

# ===================== Args Definition =======================#
args = parser.parse_args()
# vis = Visualizer(env='vg_'+args.encoder+'_'+args.tail)
# ---------- ground truth path --#
graph_path = './data/vg_pred_graph_roidb.npz'
graph_roidb = read_roidb(graph_path)
train_roidb = graph_roidb['train']
val_roidb = graph_roidb['test']
test_roidb = graph_roidb['test']
# ipdb.set_trace()
# ------------------------------------#

if args.feat_mode == 'full':
    use_vis = True
    use_sem = True
elif args.feat_mode == 'vis':
    use_vis = True
    use_sem = False
elif args.feat_mode == 'sem':
    use_vis = False
    use_sem = True
else:
    use_vis = False
    use_sem = False
    print('No feature input')
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

log = None
# Save model and meta-data. Always saves in a new folder.
if args.save_folder:
    if args.restore:
        pass
    else:
        exp_counter = 0
        save_folder = os.path.join(args.save_folder, '{}_{}_{}_exp{}'.format(args.encoder, args.feat_mode, \
                                    args.tail, exp_counter))
        while os.path.isdir(save_folder):
            exp_counter += 1
            save_folder = os.path.join(args.save_folder, '{}_{}_{}_exp{}'.format(args.encoder, args.feat_mode, \
                                    args.tail, exp_counter))
        os.mkdir(save_folder)
        meta_file = os.path.join(save_folder, 'metadata.pkl')
        model_file = os.path.join(save_folder, 'temp.pt')
        best_model_file = os.path.join(save_folder, 'encoder.pt')
        log_file = os.path.join(save_folder, 'log.txt')
        log = open(log_file, 'w')

        pickle.dump({'args': args}, open(meta_file, "wb"))
        print("save_folder: {}".format(save_folder))
else:
    print("Save_folder: {}".format(save_folder))

if args.load_folder:
    load_folder = os.path.join('./checkpoints/vg', args.encoder +'_'
            + args.feat_mode +'_'+ args.tail + '_' + args.load_folder)
    meta_file = os.path.join(load_folder, 'metadata.pkl')
    model_file = os.path.join(load_folder, 'temp.pt')
    best_model_file = os.path.join(load_folder, 'encoder.pt')
    log_file = os.path.join(load_folder, 'log_new.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
    if args.restore:
        save_folder = load_folder
else:
    load_folder = save_folder
    print("Load_folder: {}".format(load_folder))

# ===================== Model Definition ========================#
if args.encoder == 'simple':
    model = SimpleEncoder(args.hidden,
                       edge_types=args.edge_types, node_types=args.node_types,
                       do_prob=args.dropout, use_vis=use_vis, use_spatial=False, use_sem=use_sem, use_loc=args.use_loc, use_cls=args.use_cls)
elif args.encoder == 'nmp':
    model = NMPEncoder(args.hidden,
                       edge_types=args.edge_types, node_types=args.node_types, n_iter=args.n_iter,
                       do_prob=args.dropout, use_vis=use_vis, use_spatial=False, use_sem=use_sem, use_loc=args.use_loc, use_cls=args.use_cls)
if args.cuda:
    model.cuda()

# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0.0005, momentum=0, centered=False)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)
# --------------- Parameters Loader ------------------#
best_model_params = model.state_dict()
if args.restore:
    model.load_state_dict(torch.load(model_file))

# ================== Data Loader ================================#
train_loader, val_loader, test_loader = load_dataset(data_set='vg', ori_vgg=args.ori_vgg, dataset='pred', level='image',
    batch_size=args.batch_size, eval_batch_size=args.batch_size,
    shuffle=args.shuffle, feat_mode=args.feat_mode)

# ================== Loss Weights ===============================#
cls_ws_train = np.array(np.concatenate([np.ones(args.edge_types-1), [args.weight]],0), dtype=np.float32)
cls_ws_test = np.array(np.concatenate([np.ones(args.edge_types-1), [0]],0), dtype=np.float32)
cls_ws_train = torch.FloatTensor(cls_ws_train)
cls_ws_test = torch.FloatTensor(cls_ws_test)

if args.cuda:
    cls_ws_train = cls_ws_train.cuda()
    cls_ws_test = cls_ws_test.cuda()

cls_ws_train = Variable(cls_ws_train, requires_grad=False)
cls_ws_test = Variable(cls_ws_test, requires_grad=False)

# =============== iterate one epoch =====================#
def iter_one_epoch(roidb, data_loader, batch_size, is_rela=False, is_training=True):
    loss_all = []

    recall_50 = 0.0
    recall_100 = 0.0

    edge_loss_all = []
    edge_acc_all = []

    node_loss_all = []
    node_acc_all = []

    pbar = tqdm(total=len(data_loader.dataset))

    if is_rela:
        num_nodes = args.rela_num_atoms
        num_edges = num_nodes * (num_nodes - 1)
    else:
        num_nodes = args.num_atoms
        num_edges = args.num_edges

    pred_probs = np.zeros([len(data_loader.dataset), num_edges])    
    pred_cls = np.zeros([len(data_loader.dataset), num_edges]) + args.edge_types - 1
    
    for batch_idx, (data, target, node_cls, edge_feats, rel_rec, rel_send, bbox_loc, prior) in enumerate(data_loader):
        if args.cuda:
            data, target, edge_feats = data.cuda(), target.cuda(), edge_feats.cuda()
            rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            prior = prior.cuda()
            node_cls = node_cls.cuda()
            bbox_loc = bbox_loc.cuda()

        # --------- optimize ------------#
        if is_training:
            optimizer.zero_grad()

        # --------- Forward -----------#
        output, node_output = model(data, edge_feats, rel_rec, rel_send, bbox_loc)
        output = output.view(-1, args.edge_types)
        node_output = node_output.view(-1, args.node_types)

        if args.prior:
            prior = prior.view(-1, args.edge_types)
            rel_score = prior + output

        # --------- loss ----------------#
        target = target.view(-1)
        node_cls = node_cls.view(-1)

        if args.prior:
            edge_loss = F.multi_margin_loss(rel_score, target, weight=cls_ws_train, size_average=False)
            edge_count = args.edge_types / (target < args.edge_types-1).data.sum()
            loss = edge_loss * edge_count
        else:
            edge_loss = F.cross_entropy(output, target, ignore_index=args.edge_types-1)
            node_loss = F.cross_entropy(node_output, node_cls, ignore_index=args.node_types-1)
            if args.use_cls:
                loss = edge_loss + node_loss
            else:
                loss = edge_loss

        # -------- backward --------------#
        if is_training:
            # vis.plot_many_stack({'train_loss': loss.data.cpu().numpy()[0]})
            loss.backward()
            optimizer.step()

        # ============= accuracy ==============#
        # ------ edge acc -------#
        edge_acc = compute_acc(output, target, ignored_index=args.edge_types-1)
        node_acc = compute_acc(node_output, node_cls, ignored_index=args.node_types-1)
        edge_acc_all.append(edge_acc)
        node_acc_all.append(node_acc)

        loss_all.append(loss.item())
        edge_loss_all.append(edge_loss.item())
        node_loss_all.append(node_loss.item())

        # --------- save ---------------#
        output = F.softmax(output, dim=-1)
        output = output.view(-1, num_edges, args.edge_types)
        pred_prob, pred_cl = output.max(-1)

        if (batch_idx+1)*batch_size > len(data_loader.dataset):
            pred_probs[batch_idx*batch_size:] = pred_prob.data.cpu().numpy()
            pred_cls[batch_idx*batch_size:] = pred_cl.data.cpu().numpy()
        else:
            pred_probs[batch_idx*batch_size:(batch_idx+1)*batch_size] = pred_prob.data.cpu().numpy()
            pred_cls[batch_idx*batch_size:(batch_idx+1)*batch_size] = pred_cl.data.cpu().numpy()
        pbar.update(batch_size)
    pbar.close()

    if is_rela:
        pred_roidb = graph_npy2roidb(roidb, pred_probs, pred_cls, mode='rela', topk=False)
        recall_50 = eval_result(roidb, pred_roidb['pred_roidb'], 50, is_zs=False, mode='rela', topk=False, dataset='vg')
        recall_100 = eval_result(roidb, pred_roidb['pred_roidb'], 100, is_zs=False, mode='rela', topk=False, dataset='vg')
    else:
        pred_roidb = graph_npy2roidb(roidb, pred_probs, pred_cls, mode='pred', topk=False)
        recall_50 = eval_result(roidb, pred_roidb['pred_roidb'], 50, is_zs=False, mode='pred', topk=False, dataset='vg')
        recall_100 = eval_result(roidb, pred_roidb['pred_roidb'], 100, is_zs=False, mode='pred', topk=False, dataset='vg')
        
    if not is_training:
        if is_rela:
            head = 'rela_'
        else:
            head = 'pred_'
        np.savez(os.path.join(load_folder, head + 'roidb'), pred_roidb)
    return loss_all, edge_loss_all, node_loss_all, edge_acc_all, node_acc_all, recall_50, recall_100, pred_roidb

# =============== Train Op ==============================#
def train(epoch, best_val_accuracy):
    t = time.time()
    loss_train = []
    edge_loss_train = []
    node_loss_train = []
    edge_acc_train = []
    node_acc_train = []
    recall_train = 0.0

    loss_val = []
    edge_loss_val = []
    node_loss_val = []
    edge_acc_val = []
    node_acc_val = []
    recall_val = 0.0

    rela_loss_val = []
    rela_acc_val = []
    rela_recall_50 = 0.0
    rela_recall_100 = 0.0

    model.train()
    scheduler.step()
    loss_train, edge_loss_train, node_loss_train, edge_acc_train, node_acc_train, recall_train, _, pred_roidb_train = \
        iter_one_epoch(train_roidb, train_loader, args.batch_size, is_training=True)

    model.eval()
    loss_val, edge_loss_val, node_loss_val, edge_acc_val, node_acc_val, recall_val, _, pred_roidb_val = \
        iter_one_epoch(val_roidb, val_loader, args.batch_size, is_training=False)

    if args.use_cls:
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.04f}'.format(np.mean(loss_train)),
              'edge_loss_train : {:.04f}'.format(np.mean(edge_loss_train)),
              'node_loss_train : {:.04f}'.format(np.mean(node_loss_train)),
              'edge_acc_train: {:.04f}'.format(np.mean(edge_acc_train)),
              'node_acc_train: {:.04f}'.format(np.mean(node_acc_train)),
              'recall_train: {:.04f}'.format(recall_train))

        print('loss_val: {:.04f}'.format(np.mean(loss_val)),
              'edge_loss_val : {:.04f}'.format(np.mean(edge_loss_val)),
              'node_loss_val : {:.04f}'.format(np.mean(node_loss_val)),
              'edge_acc_val: {:.04f}'.format(np.mean(edge_acc_val)),
              'node_acc_val: {:.04f}'.format(np.mean(node_acc_val)),
              'recall_val: {:.04f}'.format(recall_val),
              'time: {:.4f}s'.format(time.time() - t))
    else:
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.04f}'.format(np.mean(loss_train)),
              'acc_train: {:.04f}'.format(np.mean(edge_acc_train)),
              'recall_train: {:.04f}'.format(recall_train),
              'loss_val: {:.04f}'.format(np.mean(loss_val)),
              'acc_val: {:.04f}'.format(np.mean(edge_acc_val)),
              'recall_val: {:.04f}'.format(recall_val),
              'time: {:.4f}s'.format(time.time() - t))
    torch.save(model.state_dict(), model_file)  
    if args.save_folder and recall_val > best_val_accuracy:
        torch.save(model.state_dict(), best_model_file)
        print('--------------Best model so far---------------')
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.04f}'.format(np.mean(loss_train)),
              'acc_train: {:.04f}'.format(np.mean(edge_acc_train)),
              'recall_train: {:.04f}'.format(recall_train),
              'loss_val: {:.04f}'.format(np.mean(loss_val)),
              'acc_val: {:.04f}'.format(np.mean(edge_acc_val)),
              'recall_val: {:.04f}'.format(recall_val))
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.04f}'.format(np.mean(loss_train)),
              'acc_train: {:.04f}'.format(np.mean(edge_acc_train)),
              'recall_train: {:.04f}'.format(recall_train),
              'loss_val: {:.04f}'.format(np.mean(loss_val)),
              'acc_val: {:.04f}'.format(np.mean(edge_acc_val)),
              'recall_val: {:.04f}'.format(recall_val),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return recall_val

def eval(roidb, test_loader, is_rela=False):
    t = time.time()
    loss_test = []
    edge_acc_test = []
    node_acc_test = []
    model.eval()
    if args.mode == 'eval':
        model.load_state_dict(torch.load(best_model_file))
    else:
        model.load_state_dict(torch.load(model_file))

    if is_rela:
        num_nodes = args.rela_num_atoms
        num_edges = num_nodes * (num_nodes - 1)
        batch_size = args.eval_batch_size
    else:
        num_nodes = args.num_atoms
        num_edges = args.num_edges
        batch_size = args.batch_size

    pred_probs = np.zeros([len(test_loader.dataset), num_edges])
    pred_cls = np.zeros([len(test_loader.dataset), num_edges]) + args.edge_types - 1

    pbar = tqdm(total = len(test_loader.dataset))

    for batch_idx, (data, target, node_cls, edge_feats, rel_rec, rel_send, bbox_loc, prior) in enumerate(test_loader):
        if args.cuda:
            data, target, edge_feats = data.cuda(), target.cuda(), edge_feats.cuda()
            rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            node_cls, bbox_loc = node_cls.cuda(), bbox_loc.cuda()
        
        data = data[:, :, :].contiguous()

        with torch.no_grad():
            output, node_output = model(data, edge_feats, rel_rec, rel_send, bbox_loc)
            output = output.view(-1, args.edge_types)
            node_output = node_output.view(-1, args.node_types)

        edge_acc = compute_acc(output, target.view(-1), ignored_index=args.edge_types-1)
        node_acc = compute_acc(node_output, node_cls.view(-1), ignored_index=args.node_types-1)
        edge_acc_test.append(edge_acc)
        node_acc_test.append(node_acc)

        output = F.softmax(output, dim=-1)
        output = output.view(-1 , num_edges, args.edge_types)
        pred_prob, pred_cl = output.max(-1)
        
        if (batch_idx+1)*batch_size > len(test_loader.dataset):
            pred_probs[batch_idx*batch_size:] = pred_prob.data.cpu().numpy()
            pred_cls[batch_idx*batch_size:] = pred_cl.data.cpu().numpy()
        else:
            pred_probs[batch_idx*batch_size:(batch_idx+1)*batch_size] = pred_prob.data.cpu().numpy()
            pred_cls[batch_idx*batch_size:(batch_idx+1)*batch_size] = pred_cl.data.cpu().numpy()
        pbar.update(batch_size)
    pbar.close()

    if args.use_cls:
        print('[acc] edge_acc_test: {:.04f} node_acc_test: {:.04f}'.format(np.mean(edge_acc_test), np.mean(node_acc_test)))

    # print('--------Eval-----------------')
    if is_rela:
        pred_roidb = graph_npy2roidb(roidb, pred_probs, pred_cls, mode='rela', level='image', topk=False)
        recall_50 = eval_result(roidb, pred_roidb['pred_roidb'], 50, is_zs=False, mode='rela', topk=False, dataset='vg')
        recall_100 = eval_result(roidb, pred_roidb['pred_roidb'], 100, is_zs=False, mode='rela', topk=False, dataset='vg')
        zs_recall_50 = eval_result(roidb, pred_roidb['pred_roidb'], 50, is_zs=True, mode='rela', topk=False, dataset='vg')
        zs_recall_100 = eval_result(roidb, pred_roidb['pred_roidb'], 100, is_zs=True, mode='rela', topk=False, dataset='vg')

        # np.savez(os.path.join(load_folder, 'rela_roidb'), pred_roidb)
        print('[rela_eval] recall_50: {:.4f} recall_100: {:.4f}'.format(recall_50, recall_100), file=log)
        print('[zs_rela_eval] recall_50: {:.4f} recall_100: {:.4f}'.format(zs_recall_50, zs_recall_100), file=log)
    else:
        pred_roidb = graph_npy2roidb(roidb, pred_probs, pred_cls, mode='pred', level='image', topk=False)
        recall_50 = eval_result(roidb, pred_roidb['pred_roidb'], 50, is_zs=False, mode='pred', topk=False, dataset='vg')
        recall_100 = eval_result(roidb, pred_roidb['pred_roidb'], 100, is_zs=False, mode='pred', topk=False, dataset='vg')
        zs_recall_50 = eval_result(roidb, pred_roidb['pred_roidb'], 50, is_zs=True, mode='pred', topk=False, dataset='vg')
        zs_recall_100 = eval_result(roidb, pred_roidb['pred_roidb'], 100, is_zs=True, mode='pred', topk=False, dataset='vg')
        
        np.savez(os.path.join(load_folder, 'pred_roidb'), pred_roidb)
        print('[pred_eval] recall_50: {:.4f} recall_100: {:.4f}'.format(recall_50, recall_100), file=log)
        print('[zs_pred_eval] recall_50: {:.4f} recall_100: {:.4f}'.format(zs_recall_50, zs_recall_100), file=log)

    print('recall_50: {:.4f} recall_100: {:.4f}'.format(recall_50, recall_100))
    print('[zs] recall_50: {:.4f} recall_100: {:.4f}'.format(zs_recall_50, zs_recall_100))
    return

def eval_topk(roidb, test_loader, is_rela=False, k=100):
    t = time.time()
    loss_test = []
    acc_test = []
    model.eval()
    if args.mode == 'eval':
        model.load_state_dict(torch.load(best_model_file))
    else:
        model.load_state_dict(torch.load(model_file))

    if is_rela:
        num_nodes = args.rela_num_atoms
        num_edges = num_nodes * (num_nodes - 1)
        batch_size = args.eval_batch_size
    else:
        num_nodes = args.num_atoms
        num_edges = args.num_edges
        batch_size = args.batch_size

    pred_probs = np.zeros([len(test_loader.dataset), num_edges, k])
    pred_cls = np.zeros([len(test_loader.dataset), num_edges, k]) + args.edge_types-1

    pbar = tqdm(total = len(test_loader.dataset))

    for batch_idx, (data, target, node_cls, edge_feats, rel_rec, rel_send, bbox_loc, prior) in enumerate(test_loader):
        if args.cuda:
            data, target, edge_feats = data.cuda(), target.cuda(), edge_feats.cuda()
            rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            node_cls, bbox_loc = node_cls.cuda(), bbox_loc.cuda()
        
        data = data[:, :, :].contiguous()
        with torch.no_grad():
            output, _ = model(data, edge_feats, rel_rec, rel_send, bbox_loc)
            output = output.view(-1, args.edge_types)
            output = F.softmax(output, dim=-1)
        output = output.view(-1 , num_edges, args.edge_types)

        pred_prob, pred_cl = torch.topk(output, k, dim=-1, largest=True, sorted=True)
        
        if (batch_idx+1)*batch_size > len(test_loader.dataset):
            pred_probs[batch_idx*batch_size:] = pred_prob.data.cpu().numpy()
            pred_cls[batch_idx*batch_size:] = pred_cl.data.cpu().numpy()
        else:
            pred_probs[batch_idx*batch_size:(batch_idx+1)*batch_size] = pred_prob.data.cpu().numpy()
            pred_cls[batch_idx*batch_size:(batch_idx+1)*batch_size] = pred_cl.data.cpu().numpy()
        pbar.update(batch_size)

    pbar.close()

    # print('--------Eval-----------------')
    if is_rela:
        pred_roidb = graph_npy2roidb(roidb, pred_probs, pred_cls, mode='rela', level='image', topk=True)
        recall_50 = eval_result(roidb, pred_roidb['pred_roidb'], 50, is_zs=False, mode='rela', topk=True, dataset='vg')
        recall_100 = eval_result(roidb, pred_roidb['pred_roidb'], 100, is_zs=False, mode='rela', topk=True, dataset='vg')
        zs_recall_50 = eval_result(roidb, pred_roidb['pred_roidb'], 50, is_zs=True, mode='rela', topk=True, dataset='vg')
        zs_recall_100 = eval_result(roidb, pred_roidb['pred_roidb'], 100, is_zs=True, mode='rela', topk=True, dataset='vg')
        # np.savez(os.path.join(load_folder, 'topk_rela_roidb'), pred_roidb)
        print('[rela_eval_topk] recall_50: {:.4f} recall_100: {:.4f}'.format(recall_50, recall_100), file=log)
        print('[zs_rela_eval_topk] recall_50: {:.4f} recall_100: {:.4f}'.format(zs_recall_50, zs_recall_100), file=log)
    else:
        pred_roidb = graph_npy2roidb(roidb, pred_probs, pred_cls, mode='pred', level='image', topk=True)
        recall_50 = eval_result(roidb, pred_roidb['pred_roidb'], 50, is_zs=False, mode='pred', topk=True, dataset='vg')
        recall_100 = eval_result(roidb, pred_roidb['pred_roidb'], 100, is_zs=False, mode='pred', topk=True, dataset='vg')
        zs_recall_50 = eval_result(roidb, pred_roidb['pred_roidb'], 50, is_zs=True, mode='pred', topk=True, dataset='vg')
        zs_recall_100 = eval_result(roidb, pred_roidb['pred_roidb'], 100, is_zs=True, mode='pred', topk=True, dataset='vg')
        # np.savez(os.path.join(load_folder, 'topk_pred_roidb'), pred_roidb)
        print('[pred_eval_topk] recall_50: {:.4f} recall_100: {:.4f}'.format(recall_50, recall_100), file=log)
        print('[zs_pred_eval_topk] recall_50: {:.4f} recall_100: {:.4f}'.format(zs_recall_50, zs_recall_100), file=log)

    print('recall_50: {:.4f} recall_100: {:.4f}'.format(recall_50, recall_100))
    print('[zs] recall_50: {:.4f} recall_100: {:.4f}'.format(zs_recall_50, zs_recall_100))
    return

# Train model
t_total = time.time()

if args.mode == 'whole' or args.mode == 'train':
    best_val_accuracy = -1.
    best_epoch = 0
    pbar = tqdm(total=args.epochs)

    for epoch in range(args.epochs):
        print('============= Epoch {} ==========='.format(epoch))
        val_acc = train(epoch, best_val_accuracy)
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            # print('------------- pred --------------')
            # eval(test_roidb, test_loader, is_rela=False)
            # print('------------- pred topk--------------')
            # eval_topk(test_roidb, test_loader, is_rela=False)
        pbar.update(1)
    pbar.close()

    print("======Optimization Finished!======")
    print("Best Epoch: {:04d}".format(best_epoch))
    if args.save_folder:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()
    print('------------- pred --------------')
    eval(test_roidb, test_loader, is_rela=False)
    print('------------- pred topk--------------')
    eval_topk(test_roidb, test_loader, is_rela=False)
    if log is not None:
        print(save_folder)
        log.close()
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
elif args.mode == 'eval':
    print('------------- pred --------------')
    eval(test_roidb, test_loader, is_rela=False)
    print('------------- pred topk--------------')
    eval_topk(test_roidb, test_loader, is_rela=False)
    if log is not None:
        print(load_folder)
        log.close()
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

