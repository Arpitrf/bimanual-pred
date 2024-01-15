"""
Script to test the PointNet model on the bimanual contact prediction task.
Outputs are saved in <log-dir>/eval
Run as:
    python test_bimanual_contact.py --obj tissue --log_dir pointnet_part_seg --normal --split val
    python test_bimanual_contact.py --obj tissue --log_dir pointnet_part_seg --normal --split test
"""
import argparse
import os
import os.path as osp
from pathlib import Path
from data_utils.BimanualDataLoader import PartNormalDataset, pc_normalize
from visualizer.bimanual_utils import visualize_pcl_contact
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
    parser.add_argument('--task', type=str, default='contact', help='Choose from: contact, axis')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in validation')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = osp.join('log/contact_seg', args.obj, args.log_dir)
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

    num_part = 3

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, use_cls=False, normal_channel=args.normal).cuda()
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
            seg_pred, _ = classifier(points) # dim 1xnx3
            # visualization
            points = points.transpose(2, 1).cpu().numpy() # dim 1xnx6
            logits = seg_pred[0].cpu().data.numpy() # dim nx3
            segp = np.argmax(logits, 1).astype(np.int32) # dim n
            savepath = osp.join(eval_dir, 'test.png')
            visualize_pcl_contact([segp], args.num_point, points[0,:,:3], savepath)

    elif args.split == 'val':
        
        VAL_DATASET = PartNormalDataset(root=datapath, npoints=args.num_point, task=args.task, split='val', normal_channel=args.normal)
        valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
        log_string("The number of val data is: %d" % len(VAL_DATASET))

        with torch.no_grad():
            val_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            grasp_left_ious = []
            grasp_right_ious = []
            no_grasp_ious = []
            mean_ious = []

            for batch_id, (points_in, target_in) in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points_in.size()
                points, target = points_in.float().cuda(), target_in.long().cuda()
                points = points.transpose(2, 1)
                seg_pred, _ = classifier(points)
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits, 1)

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    part_ious = [0.0 for _ in range(num_part)]
                    for l in range(num_part):
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l] = 1.0
                        else:
                            part_ious[l] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    mean_ious.append(np.mean(part_ious))
                    no_grasp_ious.append(part_ious[0])
                    grasp_left_ious.append(part_ious[1])
                    grasp_right_ious.append(part_ious[2])

            # save visualization of predictions
            points = points.transpose(2, 1).cpu().numpy()
            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target_in[i, :]
                savepath = osp.join(savedir, f'{i:02d}.png')
                visualize_pcl_contact([segl, segp], NUM_POINT, points[i,:,:3], savepath)
                
            val_metrics['accuracy'] = total_correct / float(total_seen)
            val_metrics['iou'] = np.mean(mean_ious)
            val_metrics['no_grasp_iou'] = np.mean(no_grasp_ious)
            val_metrics['grasp_left_iou'] = np.mean(grasp_left_ious)
            val_metrics['grasp_right_iou'] = np.mean(grasp_right_ious)

        log_string('val Accuracy: %f' % val_metrics['accuracy'])
        log_string('Avg mIOU: %f' % val_metrics['iou'])
        log_string('no grasp mIOU is: %.5f' % val_metrics['no_grasp_iou'])
        log_string('grasp left mIOU is: %.5f' % val_metrics['grasp_left_iou'])
        log_string('grasp right mIOU is: %.5f' % val_metrics['grasp_right_iou'])
        
        # save metrics
        with open(osp.join(eval_dir, 'metrics.txt'), 'w') as f:
            for key, value in val_metrics.items():
                f.write(f'{key}: {value}\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)
