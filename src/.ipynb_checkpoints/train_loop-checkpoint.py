import torch


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, writer):
    # print("starting training")
    model.train()
    # print("1")
    running_train_loss = 0.0
    for batch_X, batch_y in loader:
        # print('a')
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        # print('b')
        # print(f"Batch_X device: {batch_X.device}, Batch_y device: {batch_y.device}")
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    # print("2")
    avg_loss = running_train_loss / len(loader)
    # print("3")
    writer.add_scalar("Loss/train", avg_loss, epoch)
    # print("4")
    return avg_loss


def evaluate(model, loader, criterion, device, epoch = None, writer = None):
    model.eval()
    running_test_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
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
