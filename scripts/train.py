import random

import torch.optim as optim

from net import *
from test import *
from prepare import *
from graphics import *
from torch.utils.data import DataLoader


class Config:
    input_size = 1  # Размерность входных данных
    hidden_size = 64  # Размер скрытого слоя LSTM
    num_layers = 2  # Количество LSTM слоев
    output_size = 1  # Размерность выходных данных

    batch_size = 64
    num_epochs = 2
    learning_rate = 0.001


def train_model(model, train_loader, val_loader, test_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    train_losses, val_losses = [], []
    metrics = {'precision' : [], 'recall' : [], 'f1' : []}
    best_loss = float('inf')

    for epoch in range(Config.num_epochs):
        # Тренировочный цикл
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.unsqueeze(-1))
            loss = criterion(outputs, y_batch.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # Валидационный цикл
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch.unsqueeze(-1))
                loss = criterion(outputs, y_batch.unsqueeze(-1))
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), '../best_model.pth')

        train_loss = epoch_train_loss / len(train_loader)
        val_loss = avg_val_loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        p, r, f = calculate_metrics(model, test_loader)
        metrics['precision'].append(p)
        metrics['recall'].append(r)
        metrics['f1'].append(f)

        print(f'Epoch {epoch + 1}/{Config.num_epochs} | \n'
              f'\tTrain Loss: {train_loss} | '
              f'\tVal Loss: {val_loss} | \n'
              f'\tPrecision: {p} | '
              f'\tRecall: {r} | '
              f'\tF1: {f} | ')

    return train_losses, val_losses, metrics


def main():
    datasets = load_datasets('../datasets')
    train_dataset, val_dataset, test_dataset = create_dataset(datasets, sequence_length=5000, train_part=0.7, val_part=0.2)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)

    model = DenoisingLSTM(Config)

    train_loss, val_loss, metrics = train_model(model, train_loader, val_loader, test_loader)

    plot_losses(train_loss)
    plot_losses(val_loss, 'val')

    plot_sample(model, test_dataset.tensors[0], test_dataset.tensors[1], random.randint(0, len(test_dataset)))
    plot_sample(model, test_dataset.tensors[0], test_dataset.tensors[1], random.randint(0, len(test_dataset)))
    plot_sample(model, test_dataset.tensors[0], test_dataset.tensors[1], random.randint(0, len(test_dataset)))
    plot_sample(model, test_dataset.tensors[0], test_dataset.tensors[1], random.randint(0, len(test_dataset)))

    plot_metrics(metrics)


if __name__ == '__main__':
    main()