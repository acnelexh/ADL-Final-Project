# Training and evaluation functions
import torch
import torchmetrics

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, writer = None):
    # train the model for one epoch
    model.train()

    losses = []

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        
    for input_nodes, output_nodes, blocks in data_loader:
        input_nodes = input_nodes.to(device)
        output_nodes = output_nodes.to(device)
        
        # compute loss
        loss = model(x, target)
        losses.append(loss.item())
        
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # warmup lr scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()
            
    return torch.mean(torch.tensor(losses))
    
def evaluate(model, data_loader, device):
    # evaluate the model (acc for now)
    acc = torchmetrics.Accuracy()
    model.eval()
    
    for x, target in data_loader:
        pred = model(x.to(device))
        acc.update(pred, target.to(device))
    
    return acc.compute()
    
    
        
    