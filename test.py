import torch
import matplotlib.pyplot as plt

from numpy import arange


# model = DenoisingLSTM(Config)
# model.load_state_dict(torch.load('./best_model.pth', weights_only=True))
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
    plt.savefig(f'./sample_{sample_num}.png')


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


# datasets = load_datasets('./')
# train_dataset, val_dataset, test_dataset = create_dataset(datasets)
# plot_sample(model, test_dataset.tensors[0], test_dataset.tensors[1], 3)
