import argparse
import sys
import os
import time
import copy
import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torchvision import transforms

from modules.sag_resnet import sag_resnet
from modules.loss import *
from modules.utils import * 


# Training settings
parser = argparse.ArgumentParser(description='PyTorch SagNet')

# dataset
parser.add_argument('--dataset-dir', type=str, default='dataset',
                    help='home directory to dataset')
parser.add_argument('--dataset', type=str, default='pacs',
                    help='dataset name')
parser.add_argument('--sources', type=str, nargs='*',
                    help='domains for train')
parser.add_argument('--targets', type=str, nargs='*',
                    help='domains for test')

# save dir
parser.add_argument('--save-dir', type=str, default='checkpoint',
                    help='home directory to save model')
parser.add_argument('--method', type=str, default='sagnet',
                    help='method name')

# data loader
parser.add_argument('--workers', type=int, default=4,
                    help='number of workers')
parser.add_argument('--batch-size', type=int, default=32,
                    help='batch size for each source domain')
parser.add_argument('--input-size', type=int, default=256,
                    help='input image size')
parser.add_argument('--crop-size', type=int, default=224,
                    help='crop image size')
parser.add_argument('--colorjitter', type=float, default=0.4,
                    help='color jittering')

# model
parser.add_argument('--arch', type=str, default='sag_resnet',
                    help='network archiecture')
parser.add_argument('--depth', type=str, default='18',
                    help='depth of network')
parser.add_argument('--drop', type=float, default=0.5,
                    help='dropout ratio')

# sagnet
parser.add_argument('--sagnet', action='store_true', default=False,
                    help='use sagnet')
parser.add_argument('--style-stage', type=int, default=3,
                    help='stage to extract style features {1, 2, 3, 4}')
parser.add_argument('--w-adv', type=float, default=0.1,
                    help='weight for adversarial loss')

# training policy
parser.add_argument('--from-sketch', action='store_true', default=False,
                    help='training from scratch')
parser.add_argument('--lr', type=float, default=0.004,
                    help='initial learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--iterations', type=int, default=2000,
                    help='number of training iterations')
parser.add_argument('--scheduler', type=str, default='cosine',
                    help='learning rate scheduler {step, cosine}')
parser.add_argument('--milestones', type=int, nargs='+', default=[1000, 1500],
                    help='milestones to decay learning rate (for step scheduler)')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='gamma to decay learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--clip-adv', type=float, default=0.1,
                    help='grad clipping for adversarial loss')

# etc
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=10,
                    help='iterations for logging training status')
parser.add_argument('--log-test-interval', type=int, default=10,
                    help='iterations for logging test status')
parser.add_argument('--test-interval', type=int, default=100,
                    help='iterations for test')
parser.add_argument('-g', '--gpu-id', type=str, default='0',
                    help='gpu id')


def main(args):
    global status

    # Set gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # Set domains
    if args.dataset == 'pacs':
        all_domains = ['art_painting', 'cartoon', 'sketch', 'photo']
    
    if args.sources[0] == 'Rest':
        args.sources = [d for d in all_domains if d not in args.targets]
    if args.targets[0] == 'Rest':
        args.targets = [d for d in all_domains if d not in args.sources]

    # Set save dir
    save_dir = os.path.join(args.save_dir, args.dataset, args.method, ','.join(args.sources))
    print('Save directory: {}'.format(save_dir))
    os.makedirs(save_dir, exist_ok=True)

    # Set Logger
    log_path = os.path.join(save_dir, 'log.txt')
    sys.stdout = Logger(log_path)

    # Print arguments
    print('\nArguments')
    for arg in vars(args):
        print(' - {}: {}'.format(arg, getattr(args, arg)))

    # Init seed
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # Initialzie loader
    print('\nInitialize loaders...')
    init_loader()

    # Initialize model
    print('\nInitialize model...')
    init_model()

    # Initialize optimizer
    print('\nInitialize optimizers...')
    init_optimizer()
    
    # Initialize status
    src_keys = ['t_data', 't_net', 'l_c', 'l_s', 'l_adv', 'acc']
    status = OrderedDict([
        ('iteration', 0),
        ('lr', 0),
        ('src', OrderedDict([(k, AverageMeter()) for k in src_keys])),
        ('val_acc', OrderedDict([(domain, 0) for domain in args.sources])),
        ('mean_val_acc', 0),
        ('test_acc', OrderedDict([(domain, 0) for domain in args.targets])),
        ('mean_test_acc', 0),
    ])

    # Main loop
    print('\nStart training...')
    results = []
    for step in range(args.iterations):
        train(step)

        if (step + 1) % args.test_interval == 0:
            save_model(model, save_dir, 'latest')
            
            for i, domain in enumerate(args.sources):
                print('Validation: {}'.format(domain))
                status['val_acc'][domain] = test(loader_vals[i])
            for i, domain in enumerate(args.targets):
                print('Test: {}'.format(domain))
                status['test_acc'][domain] = test(loader_tgts[i])
            
            status['mean_val_acc'] = sum(status['val_acc'].values()) / len(status['val_acc'])
            status['mean_test_acc'] = sum(status['test_acc'].values()) / len(status['test_acc'])
                    
            print('Val accuracy: {:.5f} ({})'.format(status['mean_val_acc'], 
                    ', '.join(['{}: {:.5f}'.format(k, v) for k, v in status['val_acc'].items()])))
            print('Test accuracy: {:.5f} ({})'.format(status['mean_test_acc'], 
                    ', '.join(['{}: {:.5f}'.format(k, v) for k, v in status['test_acc'].items()])))
            
            results.append(copy.deepcopy(status))
            save_result(results, save_dir)


def init_loader():
    global loader_srcs, loader_vals, loader_tgts
    global num_classes

    # Set transforms
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    trans_list = []
    trans_list.append(transforms.RandomResizedCrop(args.crop_size, scale=(0.5, 1)))
    if args.colorjitter:
        trans_list.append(transforms.ColorJitter(*[args.colorjitter] * 4))
    trans_list.append(transforms.RandomHorizontalFlip())
    trans_list.append(transforms.ToTensor())
    trans_list.append(transforms.Normalize(*stats))

    train_transform = transforms.Compose(trans_list)
    test_transform  = transforms.Compose([
                        transforms.Resize(args.input_size),
                        transforms.CenterCrop(args.crop_size),
                        transforms.ToTensor(),
                        transforms.Normalize(*stats)])

    # Set datasets
    if args.dataset == 'pacs':
        from data.pacs import PACS
        image_dir = os.path.join(args.dataset_dir, args.dataset, 'images', 'kfold')
        split_dir = os.path.join(args.dataset_dir, args.dataset, 'splits')
     
        print('--- Training ---')
        dataset_srcs = [PACS(image_dir,
                             split_dir,
                             domain=domain,
                             split='train',
                             transform=train_transform)
                        for domain in args.sources]
        print('--- Validation ---')
        dataset_vals = [PACS(image_dir,
                             split_dir,
                             domain=domain,
                             split='crossval',
                             transform=test_transform)
                        for domain in args.sources]
        print('--- Test ---')
        dataset_tgts = [PACS(image_dir,
                             split_dir,
                             domain=domain,
                             split='test',
                             transform=test_transform)
                        for domain in args.targets]
        num_classes = 7
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))

    # Set loaders
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    loader_srcs = [torch.utils.data.DataLoader(
                        dataset,
                        batch_size=args.batch_size, 
                        shuffle=True, 
                        drop_last=True, 
                        **kwargs)
                   for dataset in dataset_srcs]
    loader_vals = [torch.utils.data.DataLoader(
                        dataset,
                        batch_size=int(args.batch_size * 4), 
                        shuffle=False, 
                        drop_last=False, 
                        **kwargs)
                   for dataset in dataset_vals]
    loader_tgts = [torch.utils.data.DataLoader(
                        dataset_tgt,
                        batch_size=int(args.batch_size * 4), 
                        shuffle=False, 
                        drop_last=False, 
                        **kwargs)
                   for dataset_tgt in dataset_tgts]


def init_model():
    global model
    model = sag_resnet(depth=int(args.depth),
                       pretrained=not args.from_sketch,
                       num_classes=num_classes,
                       drop=args.drop,
                       sagnet=args.sagnet,
                       style_stage=args.style_stage)

    print(model)
    model = torch.nn.DataParallel(model).cuda()


def init_optimizer():
    global optimizer, optimizer_style, optimizer_adv
    global scheduler, scheduler_style, scheduler_adv
    global criterion, criterion_style, criterion_adv
    
    # Set hyperparams
    optim_hyperparams = {'lr': args.lr, 
                         'weight_decay': args.weight_decay,
                         'momentum': args.momentum}
    if args.scheduler == 'step':
        Scheduler = optim.lr_scheduler.MultiStepLR
        sch_hyperparams = {'milestones': args.milestones,
                           'gamma': args.gamma}
    elif args.scheduler == 'cosine':
        Scheduler = optim.lr_scheduler.CosineAnnealingLR
        sch_hyperparams = {'T_max': args.iterations}
    
    # Main learning
    params = model.module.parameters()
    optimizer = optim.SGD(params, **optim_hyperparams)
    scheduler = Scheduler(optimizer, **sch_hyperparams)
    criterion = torch.nn.CrossEntropyLoss()
    
    if args.sagnet:
        # Style learning
        params_style = model.module.style_params()
        optimizer_style = optim.SGD(params_style, **optim_hyperparams)
        scheduler_style = Scheduler(optimizer_style, **sch_hyperparams)
        criterion_style = torch.nn.CrossEntropyLoss()
        
        # Adversarial learning
        params_adv = model.module.adv_params()
        optimizer_adv = optim.SGD(params_adv, **optim_hyperparams)
        scheduler_adv = Scheduler(optimizer_adv, **sch_hyperparams)
        criterion_adv = AdvLoss()


def train(step):
    global dataiter_srcs

    ## Initialize iteration
    model.train()
    
    scheduler.step()
    if args.sagnet:
        scheduler_style.step()
        scheduler_adv.step()

    ## Load data
    tic = time.time()
    
    n_srcs = len(args.sources)
    if step == 0:
        dataiter_srcs = [None] * n_srcs
    data = [None] * n_srcs
    label = [None] * n_srcs
    for i in range(n_srcs):
        if step % len(loader_srcs[i]) == 0:
            dataiter_srcs[i] = iter(loader_srcs[i])
        data[i], label[i] = next(dataiter_srcs[i])

    data = torch.cat(data)
    label = torch.cat(label)
    rand_idx = torch.randperm(len(data))
    data = data[rand_idx]
    label = label[rand_idx].cuda()
    
    time_data = time.time() - tic

    ## Process batch
    tic = time.time()

    # forward
    y, y_style = model(data)
        
    if args.sagnet:
        # learn style
        loss_style = criterion(y_style, label)
        optimizer_style.zero_grad()
        loss_style.backward(retain_graph=True)
        optimizer_style.step()
    
        # learn style_adv
        loss_adv = args.w_adv * criterion_adv(y_style)
        optimizer_adv.zero_grad()
        loss_adv.backward(retain_graph=True)
        if args.clip_adv is not None:
            torch.nn.utils.clip_grad_norm_(model.module.adv_params(), args.clip_adv)
        optimizer_adv.step()
    
    # learn content
    loss = criterion(y, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    time_net = time.time() - tic

    ## Update status
    status['iteration'] = step + 1
    status['lr'] = optimizer.param_groups[0]['lr']
    status['src']['t_data'].update(time_data)
    status['src']['t_net'].update(time_net)
    status['src']['l_c'].update(loss.item())
    if args.sagnet:
        status['src']['l_s'].update(loss_style.item())
        status['src']['l_adv'].update(loss_adv.item())
    status['src']['acc'].update(compute_accuracy(y, label))

    ## Log result
    if step % args.log_interval == 0:
        print('[{}/{} ({:.0f}%)] lr {:.5f}, {}'.format(
            step, args.iterations, 100. * step / args.iterations, status['lr'],
            ', '.join(['{} {}'.format(k, v) for k, v in status['src'].items()])))
    

def test(loader_tgt):
    model.eval()
    preds, labels = [], []
    for batch_idx, (data, label) in enumerate(loader_tgt):
        # forward
        with torch.no_grad():
            y, _ = model(data)
        
        # result
        preds += [y.data.cpu().numpy()]
        labels += [label.data.cpu().numpy()]

        # log
        if args.log_test_interval != -1 and batch_idx % args.log_test_interval == 0:
            print('[{}/{} ({:.0f}%)]'.format(
                batch_idx, len(loader_tgt), 100. * batch_idx / len(loader_tgt)))

    # Aggregate result
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    acc = compute_accuracy(preds, labels)
    return acc


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
