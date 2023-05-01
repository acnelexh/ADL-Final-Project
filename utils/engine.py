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
    
    # compute loss
    loss = model(graph, feats, labels)
    losses.append(loss.item())
    
    # compute gradient and do optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # warmup lr scheduler
    if lr_scheduler is not None:
        lr_scheduler.step()
        
    return torch.mean(torch.tensor(losses))
    
def evaluate(model, data_split, device, num_classes):
    # evaluate the model (acc for now)
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    model.eval()
    
    graph = data_split["graph"]
    val_mask = data_split["val_mask"]
    feats = data_split["feats"]
    output_labels = data_split["labels"]
    
    pred = model(graph, feats)
    acc.update(pred[val_mask], output_labels[val_mask])

    return acc.compute()
    
    
        
    