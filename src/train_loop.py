import torch

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, criterion):
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            running_test_loss += loss.item()
    return running_test_loss / len(loader)
