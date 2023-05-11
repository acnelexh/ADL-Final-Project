# Training and evaluation functions
import os
import torch
import datetime
import torchmetrics
import matplotlib.pyplot as plt
from tqdm import tqdm

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    '''
    Warmup learning rate scheduler
    args:
        optimizer: optimizer
        warmup_iters: number of warmup iterations
        warmup_factor: warmup factor
    return:
        learning rate scheduler with warmup
    '''
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def train_one_epoch(model, optimizer, graph, epoch, warm_up, args):
    '''
    Basic training function for one epoch
    args:
        model: model
        optimizer: optimizer
        graph: graph
        epoch: epoch
        warm_up: warm up, true or false
        args: arguments
    return:
        training loss for one epoch
    '''
    model.train()
    lr_scheduler = None
    
    # since batch size is always 1, this is redacted
    # if (warm_up == True) and (epoch == 0):
    #     warmup_factor = 1. / 1000
    #     warmup_iters = min(1000, len(data_loader) - 1)
    #     lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    feat = graph.ndata["feat"]
    label = graph.ndata["label"]
    edge_weight = graph.edata['w']
    
    # compute loss
    if args.edge_weight == True:
        loss = model(graph, feat, output_labels=label, edge_weight=edge_weight)
    else:
        loss = model(graph, feat, output_labels=label, edge_weight=None)

    # compute gradient and do optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # warmup lr scheduler
    if lr_scheduler is not None:
        lr_scheduler.step()
        
    return loss

def train(model,
          graph,
          optimizer,
          lr_scheduler,
          write_dirs,
          args,
          writer):
    """
    Train the model
    input:
        model: model
        graph: graph
        optimizer: optimizer
        lr_scheduler: learning rate scheduler
        write_dirs: directory to write logs and models
        args: arguments
        writer: tensorboard writer
    return:
        None
    """
    # initialize
    device = args.device
    warm_up = args.warm_up
    eval_acc = dict()
    train_acc = dict()
    losses = dict()

    model.to(device)
    # train the model
    for epoch in tqdm(range(args.epochs)):
        # train the model for one epoch===========================================================
        time = datetime.datetime.now()
        loss = train_one_epoch(model, optimizer, graph, epoch, warm_up, args)
        losses[epoch] = loss.item()
        epoch_time = datetime.datetime.now() - time
        lr_scheduler.step()
        
        # save the model according to the interval================================================
        if (epoch + 1) % args.save_model_interval == 0 or (epoch + 1) == args.epochs:
            torch.save(model, os.path.join(write_dirs, 'model_{}.pth'.format(epoch)))
            torch.save(model, os.path.join(write_dirs, 'model_last.pth'))
            
        # evaluate the model on the training set===================================================
        if args.edge_weight == True:
            acc = evaluate(model, graph, split_mask='train_mask', e_weight=graph.edata['w']).item()
        else:
            acc = evaluate(model, graph, split_mask='train_mask').item()
        train_acc[epoch] = acc
        
        # evaluate the model on the validation set=================================================
        time = datetime.datetime.now()
        if args.edge_weight:
            acc = evaluate(model, graph, split_mask='val_mask', e_weight=graph.edata['w']).item()
        else:
            acc = evaluate(model, graph, split_mask='val_mask').item()
        eval_time = datetime.datetime.now() - time
        eval_acc[epoch] = acc
        
        # save the best model======================================================================
        if acc >= max(eval_acc.values()):
            torch.save(model, os.path.join(write_dirs, 'model_best.pth'))
            
        # log the evaluation=======================================================================
        if (epoch + 1) % args.log_epochs == 0 or (epoch + 1) == args.epochs:
            with open(os.path.join(write_dirs, 'log.txt'), 'a') as f:
                f.write(f'Epoch: {epoch},\t train loss: {round(loss.item(), 4)}, train time: {epoch_time}\n')
                f.write(f'Epoch: {epoch},\t  validation accuracy: {round(acc, 4)},\t eval time: {eval_time}\n')
                
        # tensorboard writer=======================================================================
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('Accuracy/val', acc, epoch)
        
    # plot the accuracy and loss curves============================================================
    plot_loss(losses, write_dirs)
    plot_acc(train_acc, eval_acc, write_dirs)
    
    # load the best model and test=================================================================
    best_model = torch.load(os.path.join(write_dirs, 'model_best.pth'))
    if args.edge_weight:
        acc = evaluate(best_model, graph, split_mask='test_mask', e_weight=graph.edata['w']).item()
    else:
        acc = evaluate(best_model, graph, split_mask='test_mask').item()
     
    # log the test accuracy========================================================================
    with open(os.path.join(write_dirs, 'log.txt'), 'a') as f:
        f.write(f'Test accuracy: {round(acc, 4)}\n')
    writer.add_scalar('Accuracy/test', acc, epoch)

def plot_loss(loss, write_dirs):
    '''
    Plot the loss curve and save it to the write directory
    '''
    plt.clf()
    plt.plot(loss.keys(), loss.values(), label='train')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.savefig(os.path.join(write_dirs, 'loss.png'))
    plt.close()

def plot_acc(train_acc, eval_acc, write_dirs):
    '''
    Plot train and eval accuracy curves and save it to the write directory
    '''
    plt.clf()
    plt.plot(train_acc.keys(), train_acc.values(), label='train')
    plt.plot(eval_acc.keys(), eval_acc.values(), label='eval')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.savefig(os.path.join(write_dirs, 'acc.png'))
    plt.close()

def evaluate(model, graph, split_mask='val_mask', e_weight=None):
    '''
    Evaluate the model on the split mask
    args:
        model: model
        graph: graph
        split_mask: mask to evaluate on, train_mask, val_mask, test_mask
        e_weight: edge weight (optional and model dependent)
    return:
        acc: accuracy on the split mask set of nodes
    '''
    model.eval()
    
    mask = graph.ndata[split_mask]
    feats = graph.ndata["feat"]
    output_labels = graph.ndata["label"]
    
    pred = model(graph, feats, edge_weight=e_weight)
    
    # calcualte accuracy
    pred = pred[mask].argmax(dim=1)
    labels = output_labels[mask]
    acc = (pred == labels).sum().float() / len(pred)
    
    return acc
    
    
        
    