"""
Script to train a model for contact segmentation on the bimanual dataset.
Run as:
    python train_bimanual_contact.py --obj tissue --model pointnet_part_seg --normal --log_dir bimanual_contact_pointnet_part_seg --gpu 0 --epoch 1001
"""
import argparse
import os
import os.path as osp
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from data_utils.BimanualDataLoader import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--data_dir', type=str, default='data/bimanual', help='data directory')
    parser.add_argument('--obj', type=str, default='tissue', help='object name')
    parser.add_argument('--task', type=str, default='contact', help='Choose from: contact, axis')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=501, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    viz_dir = exp_dir.joinpath('viz/')
    viz_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    datapath = osp.join(args.data_dir, args.obj)

    TRAIN_DATASET = PartNormalDataset(root=datapath, npoints=args.npoint, task=args.task, split='train', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=datapath, npoints=args.npoint, task=args.task, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_part = 3

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model(num_part, use_cls=False, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            # TODO: verify augmentations
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            grasp_left_ious = []
            grasp_right_ious = []
            no_grasp_ious = []
            mean_ious = []

            classifier = classifier.eval()

            for batch_id, (points_in, target_in) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points_in.size()
                points, target = points_in.float().cuda(), target_in.long().cuda()
                # print('points: ', points.shape)
                # print('target: ', target.shape)
                points = points.transpose(2, 1)
                seg_pred, _ = classifier(points)
                # print('seg_pred: ', seg_pred.shape)
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                # print('cur_pred_val: ', cur_pred_val.shape)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    logits = cur_pred_val_logits[i, :, :]
                    # print('logits: ', logits.shape)
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
            if epoch % 50 == 0:
                points = points.transpose(2, 1).cpu().numpy()
                savedir = viz_dir.joinpath(f'{epoch:03d}')
                savedir.mkdir(exist_ok=True)
                for i in range(cur_batch_size):
                    fig, ax = plt.subplots(1,2)
                    segp = cur_pred_val[i, :]
                    segl = target_in[i, :]
                    for idx, seg in enumerate([segl, segp]):
                        # org_colors: Nx3, seg: N
                        org_colors = np.zeros((NUM_POINT, 3))
                        org_colors[seg == 0] = [0.0, 0.0, 1.0]  # no contact: blue
                        org_colors[seg == 1] = [0.0, 1.0, 0.0]  # left contact: green
                        org_colors[seg == 2] = [1.0, 0.0, 0.0]  # right contact: red
                        o3d_pcl = o3d.geometry.PointCloud()
                        o3d_pcl.points = o3d.utility.Vector3dVector(points[i,:,:3])
                        o3d_pcl.colors = o3d.utility.Vector3dVector(org_colors)
                        
                        vis = o3d.visualization.Visualizer()
                        vis.create_window(visible=False)
                        vis.add_geometry(o3d_pcl)
                        vis.update_geometry(o3d_pcl)
                        vis.poll_events()
                        vis.update_renderer()
                        img = vis.capture_screen_float_buffer(True)
                        vis.destroy_window()
                        ax[idx].imshow(np.asarray(img))
                    plt.savefig(savedir.joinpath(f'{i:02d}.png'))

            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['iou'] = np.mean(mean_ious)
            test_metrics['no_grasp_iou'] = np.mean(no_grasp_ious)
            test_metrics['grasp_left_iou'] = np.mean(grasp_left_ious)
            test_metrics['grasp_right_iou'] = np.mean(grasp_right_ious)

        log_string('Epoch %d test Accuracy: %f  mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['iou']))
        if (test_metrics['iou'] >= best_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'test_iou': test_metrics['iou'],
                'test_no_grasp_iou': test_metrics['no_grasp_iou'],
                'test_grasp_left_iou': test_metrics['grasp_left_iou'],
                'test_grasp_right_iou': test_metrics['grasp_right_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['iou'] > best_iou:
            best_iou = test_metrics['iou']
        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best avg mIOU is: %.5f' % best_iou)
        log_string('Best no grasp mIOU is: %.5f' % test_metrics['no_grasp_iou'])
        log_string('Best grasp left mIOU is: %.5f' % test_metrics['grasp_left_iou'])
        log_string('Best grasp right mIOU is: %.5f' % test_metrics['grasp_right_iou'])
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
