import torch
import matplotlib.pyplot as plt

from numpy import arange
from sklearn.metrics import precision_score, recall_score, f1_score



# model = DenoisingLSTM(Config)
# model.load_state_dict(torch.load('../best_model.pth', weights_only=True))
# model.eval()


def plot_sample(model, test_x, test_y, sample_num=0):
    """
        Выводит на экран графики 1 сэмпла, выбранного номера: входной ряд, целевой ряд, отфильтрованный нейронной сетью

    :param model: Модель сети для фильтрации
    :param test_x: Тензор всей тестовой выборки входных данных
    :param test_y: Тензор всей тестовой выборки целевых данных
    :param sample_num: Номер сэмпла из выборок

    :return:
    """
    test_x = test_x[sample_num]
    test_y = test_y[sample_num]
    test_pred = denoise_signal(model, test_x)

    plt.figure(figsize=(15, 6))
    time = arange(0, len(test_x))
    plt.title(f'Графики {sample_num} сэмпла тестовой выборки')
    plt.plot(time, test_x, label='Шумные данные (X)', alpha=1, linewidth=1)
    plt.plot(time, test_y, label='Целевая модель (y)', alpha=1, linewidth=1.5)
    plt.plot(time, test_pred, label='Отфильтрованная модель (pred)', alpha=1, linewidth=1.5)
    plt.xlabel('time', fontsize=14)
    plt.ylabel('signal', fontsize=14)
    plt.legend(loc='upper right', prop={'size': 14})
    plt.savefig(f'../graphics/sample_{sample_num}.png')


def denoise_signal(model, input_signal):
    """
        Выполняет фильтрацию временного ряда

    :param model: Предобученная модель
    :param input_signal: Сэмпл для фильтраци (Тензор)

    :return:
    """
    input_tensor = input_signal.unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        output = model(input_tensor)
        output = (output > 0.5).float()

    return output.squeeze()


def calculate_metrics(model, test_loader):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.unsqueeze(-1))
            all_preds.append(outputs.squeeze())
            all_targets.append(y_batch)

    predictions = torch.cat(all_preds).flatten()
    targets = torch.cat(all_targets).flatten()
    y_pred_binary = (predictions >= 0.5).int()

    # errors = (predictions - targets).abs()
    # bin_preds = (errors < threshold).float()
    # bin_targets = torch.ones_like(bin_preds)

    precision = precision_score(targets, y_pred_binary)
    recall = recall_score(targets, y_pred_binary)
    f1 = f1_score(targets, y_pred_binary)

    # precision = recall_score(bin_targets, bin_preds)
    # recall = recall_score(bin_targets, bin_preds)
    # f1 = f1_score(bin_targets, bin_preds)

    # print(f'Precision: {precision}')
    # print(f'Recall: {recall}')
    # print(f'F1-Score: {f1}')

    return precision, recall, f1


# datasets = load_datasets('../datasets')
# train_dataset, val_dataset, test_dataset = create_dataset(datasets)
# test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)
# calculate_metrics(model, test_loader)
# plot_sample(model, test_dataset.tensors[0], test_dataset.tensors[1], 0)
