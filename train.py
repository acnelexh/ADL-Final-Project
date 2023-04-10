# main function for training
import argparse
import torch
import torch.nn as nn
import datetime
import os
from utils import train_one_epoch, evaluate, get_dataloader, fetch_model_fn

def get_args():
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--dataset', default='./data/exported-traced-adjacencies', help='dataset path')
    parser.add_argument('--model', default='SimpleGNN', help='model name')
    
    # training options
    parser.add_argument('--save_dir', default='./runs',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    
    # training parameters
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr-scheduler', default="multisteplr", help='the lr scheduler (default: multisteplr)')
    parser.add_argument('--lr-step-size', default=8, type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma (multisteplr scheduler only)')
    parser.add_argument('--lr_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # dataloader parameters
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    
    # logging and saving
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--save_model_interval', type=int, default=1)
    
    args = parser.parse_args()
    
    return args
    

def train(model,
          dataloader,
          optimizer,
          lr_scheduler,
          args):
    """
    Train the model
    input:
        args: the arguments
    return:
        None
    """
    print_freq = args.print_freq
    device = args.device
    eval_acc = dict()
    
    for epoch in range(args.epochs):
        # train the model
        time = datetime.datetime.now()
        loss = train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq, writer = None)
        epoch_time = datetime.datetime.now() - time
        lr_scheduler.step()
        # save the model
        if args.output_dir:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            torch.save(checkpoint, os.path.join(args.output_dir, 'model_last.pth'))
        # save the log
        with open(os.path.join(args.log_dir, 'log.txt'), 'a') as f:
            f.write(f'Epoch: {epoch}, train loss: {loss}, train time: {epoch_time}')
        
        if (epoch + 1) % args.log_epochs == 0 or (epoch + 1) == args.epochs:
            # evaluate the model
            time = datetime.datetime.now()
            acc = evaluate(model, dataloader, device, epoch, print_freq, writer = None)
            eval_time = datetime.datetime.now() - time
            with open(os.path.join(args.log_dir, 'log.txt'), 'a') as f:
                f.write(f'Epoch: {epoch}, validation accuracy: {acc}, eval time: {eval_time}')
            eval_acc[epoch] = acc
            # save the best model
            if args.output_dir:
                if acc == max(eval_acc.values()):
                    torch.save(checkpoint, os.path.join(args.output_dir, 'model_best.pth'))
            

def main(args):
    # create the experiment directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    # create model and optimizer
    # hardcode for now
    model = fetch_model_fn(args)(2, 1024, 6223)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_decay)
    if args.lr_scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
                            "are supported.".format(args.lr_scheduler))
    
    dataloader = get_dataloader(args)
    
    train(model, dataloader, optimizer, lr_scheduler, args)
    

if __name__ == "__main__":
    args = get_args()
    main(args)
    