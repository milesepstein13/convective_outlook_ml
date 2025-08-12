import torch


def train_one_epoch(model, loader, optimizer, criterion, epoch, writer):
    model.train()
    running_train_loss = 0.0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    avg_loss = running_train_loss / len(loader)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    return avg_loss


def evaluate(model, loader, criterion, epoch = None, writer = None):
    model.eval()
    running_test_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            running_test_loss += loss.item()

            all_preds.append(outputs.cpu())
            all_targets.append(batch_y.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)                           
                               
    avg_loss = running_test_loss / len(loader)
    if epoch is not None:
        writer.add_scalar("Loss/val", avg_loss, epoch)
    return avg_loss, all_preds, all_targets
