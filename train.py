# main function for training
import argparse
import torch
import torch.nn as nn
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from utils import train_one_epoch, evaluate, fetch_model_fn, get_hemibrain_split

def get_args():
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--dataset', default='./data/exported-traced-adjacencies', help='dataset path')
    parser.add_argument('--model', default='SimpleGNN', help='model name')
    parser.add_argument('--num_classes', type=int, default=6224, help='number of classes')
    
    # training options
    parser.add_argument('--output_dir', default='./runs',
                        help='Directory to save the model')
    parser.add_argument('--tensorboard_log_dir', default='./tensorboard_log', help='tensorboard log directory')
    
    # training parameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_scheduler', default="multisteplr", help='the lr scheduler (default: multisteplr)')
    parser.add_argument('--lr_step_size', default=-1, type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr_steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma (multisteplr scheduler only)')
    parser.add_argument('--lr_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warm_up', help='warmup the learning rate')
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=64, help='hidden dimension')
    
    
    # dataloader parameters
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    
    # logging and saving
    parser.add_argument('--print_freq', default=20, type=int, help='print frequency')
    parser.add_argument('--save_model_interval', type=int, default=10)
    parser.add_argument('--log_epochs', type=int, default=1)
    
    args = parser.parse_args()
    
    return args
    

def train(model,
          data_split,
          optimizer,
          lr_scheduler,
          write_dirs,
          args,
          writer):
    """
    Train the model
    input:
        args: the arguments
    return:
        None
    """
    device = args.device
    warm_up = args.warm_up
    eval_acc = dict()
    num_classes = args.num_classes
    
    model.to(device)
    for epoch in tqdm(range(args.epochs)):
        # train the model
        time = datetime.datetime.now()
        loss = train_one_epoch(model, optimizer, data_split, device, epoch, warm_up)
        epoch_time = datetime.datetime.now() - time
        lr_scheduler.step()
        # save the model
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'args': args,
            'epoch': epoch
        }
        if (epoch + 1) % args.save_model_interval == 0:
            torch.save(checkpoint, os.path.join(write_dirs, 'model_{}.pth'.format(epoch)))
            torch.save(checkpoint, os.path.join(write_dirs, 'model_last.pth'))
        # save the log
        with open(os.path.join(write_dirs, 'log.txt'), 'a') as f:
            f.write(f'Epoch: {epoch},\t train loss: {round(loss.item(), 4)}, train time: {epoch_time}\n')
        if (epoch + 1) % args.log_epochs == 0 or (epoch + 1) == args.epochs:
            # evaluate the model
            time = datetime.datetime.now()
            acc = evaluate(model, data_split, device, num_classes).item()
            eval_time = datetime.datetime.now() - time
            with open(os.path.join(write_dirs, 'log.txt'), 'a') as f:
                f.write(f'Epoch: {epoch},\t  validation accuracy: {round(acc, 4)},\t eval time: {eval_time}\n')
            eval_acc[epoch] = acc
            # save the best model
            if acc > max(eval_acc.values()):
                torch.save(checkpoint, os.path.join(write_dirs, 'model_best.pth'))
        
        # tensorboard stuff
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('Accuracy/val', acc, epoch)
            
            

def main(args):
    if not os.path.exists(args.tensorboard_log_dir):
        os.makedirs(args.tensorboard_log_dir)
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    
    # create the experiment directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    write_dir = args.output_dir + '/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(os.path.join(write_dir))
    
    # save training parameters
    with open(os.path.join(write_dir, 'train.sh'), 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('python train.py \\\n')
        for key, value in vars(args).items():
            if type(value) == list:
                value = ' '.join([str(v) for v in value])
            f.write("   --{} {} \\\n".format(key, value))
    
    # create model and optimizer
    # hardcode for now
    model = fetch_model_fn(args)(2, args.hidden_dim, args.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_decay)
    if args.lr_scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'step_lr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
                            "are supported.".format(args.lr_scheduler))
    
    data_split = get_hemibrain_split(args)
    
    train(model, data_split, optimizer, lr_scheduler, write_dir, args, writer)
    

if __name__ == "__main__":
    args = get_args()
    main(args)
    