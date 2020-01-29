"""

"""
import torch
import torch.nn as nn
from tqdm import tqdm
from model import RNN
from dataloader import *
import config
import time

CUDA = torch.cuda.is_available()
np.random.seed(config.random_seed)
torch.manual_seed(config.random_seed)

if CUDA:
    gpu_cpu = torch.device('cuda')
    torch.cuda.manual_seed(config.random_seed)
else:
    torch.device('cpu')


def get_long_tensor(x):
    return torch.LongTensor(x).to(gpu_cpu)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(train: Examples, model: RNN, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    count = 0
    model.train()
    for x, y, z in batch(train.shuffled(), config.batch_size):
        x, y, z = get_long_tensor(x), get_long_tensor(y).float(), get_long_tensor(z)

        optimizer.zero_grad()
        if config.setting == 'RNN':
            predictions = model(x).squeeze(1)
        else:
            predictions = model(x, z).squeeze(1)

        loss = criterion(predictions, y)

        acc = binary_accuracy(predictions, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        count += 1
    return epoch_loss / count, epoch_acc / count


def evaluate(eval: Examples, model: RNN, criterion):
    epoch_loss = 0
    epoch_acc = 0
    count = 0
    model.eval()

    with torch.no_grad():
        for x, y, z in batch(eval, config.batch_size):
            x, y, z = get_long_tensor(x), get_long_tensor(y).float(), get_long_tensor(z)
            if config.setting == 'RNN':
                predictions = model(x).squeeze(1)
            else:
                predictions = model(x, z).squeeze(1)

            loss = criterion(predictions, y)

            acc = binary_accuracy(predictions, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            count += 1
    return epoch_loss / count, epoch_acc / count


def main(dl: DataLoader, model: RNN):
    prev_best = 0
    patience = 0
    decay = 0
    lr = config.lr

    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(config.max_epochs)):
        start_time = time.time()
        train_loss, train_acc = train(dl.train_examples, model, optimizer, criterion)
        dev_loss, dev_acc = evaluate(dl.dev_examples, model, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}')
        print(f'\t  Dev Loss: {dev_loss:.3f} |   Dev Acc: {dev_acc:.2f}')

        if dev_acc <= prev_best:
            patience += 1
            if patience == 3:
                lr *= 0.5
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                tqdm.write('Dev accuracy did not increase in 3 epochs, halfing the learning rate')
                patience = 0
                decay += 1
        else:
            prev_best = dev_acc
            print('Save the best model')
            model.save()

        if decay >= 3:
            print('Evaluating model on test set')
            model.load()
            print('Load the best model')

            test_loss, test_acc = evaluate(dl.test_examples, model, criterion)

            print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
            break


if __name__ == "__main__":
    dl = DataLoader(config)
    model = RNN(config, dl, gpu_cpu)
    main(dl, model)
