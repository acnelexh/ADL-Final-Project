# main function for training
import os
import torch
import argparse
import datetime
from torch.utils.tensorboard import SummaryWriter
from utils import train, fetch_model, get_hemibrain_split

def get_args():
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--dataset', default='./data/exported-traced-adjacencies', help='dataset path')
    parser.add_argument('--model', default='SimpleGNN', help='model name')
    #parser.add_argument('--num_classes', type=int, default=6224, help='number of classes')
    
    # training options
    parser.add_argument('--output_dir', default='./runs',
                        help='Directory to save the model')
    parser.add_argument('--tensorboard_log_dir', default='./tensorboard_log', help='tensorboard log directory')
    
    # training parameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_scheduler', default="multisteplr", help='the lr scheduler (default: multisteplr)')
    parser.add_argument('--lr_step_size', default=20, type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr_steps', default=[16, 22, 40], nargs='+', type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma (multisteplr scheduler only)')
    parser.add_argument('--lr_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warm_up', help='warmup the learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    
    # model parameters
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=[512, 512], help='hidden dimension')
    parser.add_argument('--input_dim', type=int, default=6, help='input dimension')
    parser.add_argument('--label_embed_dim', type=int, default=256, help='label embedding dimension')
    
    # experiment parameters
    parser.add_argument('--random_split', default=False, help='randomly/topologically split the dataset')
    parser.add_argument('--label_weight', type=float, default=5.0, help='weight for CE loss for nodes with label embedding')
    parser.add_argument('--unlabel_weight', type=float, default=1.0, help='weight for CE loss for nodes without label embedding')
    parser.add_argument('--normalize', default=True, help='normalize the features')
    parser.add_argument('--proportion', type=float, default=0.1, help='proportion of nodes with label embedding')
    parser.add_argument('--sample_method', default='random', help='sample method for label embedding, options: [random, degree, locality, label]')
    parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads for GAT model')
    parser.add_argument('--edge_weight', type=bool, default=True, help='use edge weight')
    parser.add_argument('--topological_feature', default=False, help='use topological feature only')
    parser.add_argument('--few_shot', default=True, help='few shot learning with label embedding')
    parser.add_argument('--class_balance_loss', default=False, help='use class balance loss, if this is true then not using label embedding')
    
    # dataloader parameters
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--data_split', type=float, default=[0.6, 0.2, 0.2], nargs='+', help='train/val/test split')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    
    # logging and saving
    parser.add_argument('--print_freq', default=20, type=int, help='print frequency')
    parser.add_argument('--save_model_interval', type=int, default=10)
    parser.add_argument('--log_epochs', type=int, default=1)
    
    args = parser.parse_args()
    
    return args
            

def main(args):
    torch.random.manual_seed(args.seed)
    
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
    graph, num_classes = get_hemibrain_split(args)
    args.num_classes = num_classes
    args.input_dim = graph.ndata['feat'].shape[1]
    # hardcode for now
    model = fetch_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_decay)
    if args.lr_scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'steplr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
                            "are supported.".format(args.lr_scheduler))
    
    train(model, graph, optimizer, lr_scheduler, write_dir, args, writer)
    
    # load best model and test
    #model = torch.load(os.path.join(write_dir, 'model_best.pth.tar'))
    
    

if __name__ == "__main__":
    args = get_args()
    main(args)
    