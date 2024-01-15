"""
Script to test the PointNet model on the bimanual axis prediction task.
Outputs are saved in <log-dir>/eval
Run as:
    python test_bimanual_axis.py --obj tissue --log_dir pointnet_reg --normal --split val
    python test_bimanual_axis.py --obj tissue --log_dir pointnet_reg --normal --split test
"""
import argparse
import os
import os.path as osp
from pathlib import Path
from data_utils.BimanualDataLoader import PartNormalDataset, pc_normalize
from visualizer.bimanual_utils import visualize_pcl_axis
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--data_dir', type=str, default='data/bimanual', help='data directory')
    parser.add_argument('--obj', type=str, default='tissue', help='object to evaluate')
    parser.add_argument('--split', type=str, default='val', help='Choose from: val, test')
    parser.add_argument('--task', type=str, default='axis', help='Choose from: contact, axis')
    parser.add_argument('--use_q', action='store_true', default=False, help='use q in axis prediction')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in validation')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    parser.add_argument('--mat_diff_loss_scale', type=float, default=0.001, help='weight for matching different loss')
    parser.add_argument('--axis_loss_scale', type=float, default=1.0, help='weight for axis loss')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = osp.join('log/axis_reg', args.obj, args.log_dir)
    eval_dir = osp.join(experiment_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    savedir = osp.join(eval_dir, 'viz')
    os.makedirs(savedir, exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % eval_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    datapath = osp.join(args.data_dir, args.obj)

    if args.use_q:
        k = 6
    else:
        k = 3

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(k, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss(mat_diff_loss_scale=args.mat_diff_loss_scale, axis_loss_scale=args.axis_loss_scale).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    if args.split == 'test':

        with torch.no_grad():
            data = np.loadtxt(os.path.join(datapath, 'test.csv'), delimiter=',').astype(np.float32) # dim nx6: [x,y,z,nx,ny,nz]
            if not args.normal:
                points = data[:, 0:3]
            else:
                points = data[:, 0:6]
            points[:, 0:3] = pc_normalize(points[:, 0:3])
            choice = np.random.choice(len(points), args.num_point, replace=True)
            points = points[choice, :]
            points = torch.from_numpy(points).float().cuda()
            points = points.transpose(1, 0).unsqueeze(0) # dim 1x6xn
            axis_pred, _ = classifier(points) # dim 1x3 or 1x6
            # visualization
            points = points.transpose(2, 1).cpu().numpy() # dim 1xnx6
            axisp = axis_pred[0].cpu().data.numpy() # dim 3 or 6
            savepath = osp.join(eval_dir, 'test.png')
            visualize_pcl_axis([axisp], args.num_point, points[0,:,:3], savepath)

    elif args.split == 'val':
        
        VAL_DATASET = PartNormalDataset(root=datapath, npoints=args.num_point, task=args.task, split='val', normal_channel=args.normal, use_q=args.use_q)
        valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
        log_string("The number of val data is: %d" % len(VAL_DATASET))

        with torch.no_grad():
            val_metrics = {}
            val_losses = {'total': [], 'axis': [], 'mat_diff': []}

            classifier = classifier.eval()

            for batch_id, (points, target) in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, target = points.float().cuda(), target.float().cuda()
                points = points.transpose(2, 1)
                axis_pred, trans_feat = classifier(points)
                loss_dict = criterion(axis_pred, target, trans_feat)
                val_losses['total'].append(loss_dict['total'].item())
                val_losses['axis'].append(loss_dict['axis'].item())
                val_losses['mat_diff'].append(loss_dict['mat_diff'].item())
                
           # save visualization of predictions
            points = points.transpose(2, 1).cpu().numpy()
            axis_pred = axis_pred.cpu().data.numpy()
            target = target.cpu().data.numpy()
            for i in range(cur_batch_size):
                axisp = axis_pred[i, :]
                axisl = target[i, :]
                savepath = osp.join(savedir, f'{i:02d}.png')
                visualize_pcl_axis([axisl, axisp], NUM_POINT, points[i,:,:3], savepath, use_q=args.use_q)

            val_metrics['total_loss'] = np.mean(val_losses['total'])
            val_metrics['axis_loss'] = np.mean(val_losses['axis'])
            val_metrics['mat_diff_loss'] = np.mean(val_losses['mat_diff'])

        log_string('Val loss\t total: {:.4f} axis: {:.4f} mat diff: {:.4f}' .format(
            val_metrics['total_loss'], val_metrics['axis_loss'], val_metrics['mat_diff_loss']))

        # save metrics
        with open(osp.join(eval_dir, 'metrics.txt'), 'w') as f:
            for key, value in val_metrics.items():
                f.write(f'{key}: {value}\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)
