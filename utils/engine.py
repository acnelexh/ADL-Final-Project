# Training and evaluation functions
import os
import torch
import datetime
import torchmetrics
from tqdm import tqdm

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def train_one_epoch(model, optimizer, data_split, device, epoch, warm_up, writer = None):
    # train the model for one epoch
    model.train()

    losses = []

    lr_scheduler = None
    # if (warm_up == True) and (epoch == 0):
    #     warmup_factor = 1. / 1000
    #     warmup_iters = min(1000, len(data_loader) - 1)
    #     lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    graph = data_split["graph"]
    feats = data_split["feats"]
    labels = data_split["labels"]
    e_weight = graph.edata['w']
    
    # compute loss
    # TODO make this more generaliziable?
    loss = model(graph, feats, e_weight, labels)
    #loss = model(graph, feats, labels)
    losses.append(loss.item())
    
    # compute gradient and do optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # warmup lr scheduler
    if lr_scheduler is not None:
        lr_scheduler.step()
        
    return torch.mean(torch.tensor(losses))

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
        if (epoch + 1) % args.save_model_interval == 0 or (epoch + 1) == args.epochs:
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
        
def evaluate(model, data_split, device, num_classes):
    # evaluate the model (acc for now)
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    model.eval()
    
    graph = data_split["graph"]
    val_mask = data_split["val_mask"]
    feats = data_split["feats"]
    output_labels = data_split["labels"]
    e_weight = graph.edata['w']
    
    pred = model(graph, feats, e_weight)
    acc.update(pred[val_mask], output_labels[val_mask])

    return acc.compute()
    
    
        
    