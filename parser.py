import os
from pickle import UnpicklingError
from typing import Optional

import matplotlib.pyplot as plt
import torch
import numpy as np


def load_xy_from_pt(path) -> dict:
    """
        Функция загружает шумные и целевые данные в виде словаря из файла .pt

    :param path: Путь к файлу .pt

    :return data: Словарь с шумными данными 'X' и целевыми данными 'y'
    """

    try:
        data = torch.load(path)
        print(f"Файл {path} загружен.")
    except UnpicklingError:
        print(f"Не удалось загрузить данные. Неверно указан путь к файлу {path}.")
        return

    if 'X' not in data or 'y' not in data:
        raise ValueError("Файл не содержит необходимых ключей 'X' и 'y'.")

    return data


def plot_series(data, sample_num: int = 155, start_idx: int = 0, num_indexes: int = 8000, fig_size: Optional[tuple[float]] = (15., 6.)):
    """
    Функция отображает графики шумных и целевых данных 1 сэмпла

    :param data: Данные в виде словаря с двумя тензорами
    :param sample_num: Номер сэмпла из выборки, который необходимо отобразить
    :param start_idx: Индекс временной метки, с которой начинать график
    :param num_indexes: Количество временных меток со стартового
    :param fig_size: Размер графика

    :return:
    """
    X = data['X'].numpy()
    y = data['y'].numpy()

    # Проверка размерностей данных
    if X.ndim > 2 or y.ndim > 2:
        raise ValueError("Данные должны быть 1D или 2D тензорами")

    if X.ndim == 2:
        X = X[:, sample_num]
    if y.ndim == 2:
        y = y[:, sample_num]

    end_idx = min(start_idx + num_indexes, len(X))
    X_plot = X[start_idx:end_idx]
    y_plot = y[start_idx:end_idx]
    time = np.arange(start_idx, end_idx)

    plt.figure(figsize=fig_size)
    plt.title(f"Графики №{sample_num}")
    plt.plot(time, X_plot, label='Шумные данные (X)', alpha=1, linewidth=1)
    plt.plot(time, y_plot, label='Целевая модель (y)', alpha=1, linewidth=1.5)

    plt.show()


def create_dataset(datasets: list[dict],
                   sequence_length: int = 2000,
                   train_part: int = 0.7,
                   val_part: int = 0.2):
    """
        Функция объединяет несколько датасетов в 1 общий и разделяет его на тренировочную, валидационную и тестовую выборку
        согласно входным параметрам. Если число временных меток в датасете не делится нацело на требуемую длину секции,
        то датасет обрезается и часть данных (последний временной ряд, который меньше длины секции) в обучении не
        используется

    :param datasets: Список со словарями датасетов
    :param sequence_length: Длина секции, 1 сэмпла
    :param train_part: Процент сэмплов для обучения
    :param val_part: Процент сэмплов для валидации
    :param test_part: Процент сэмплов для теста

    :return:
    """
    xs, ys = None, None
    for dataset in datasets:
        x, y = torch.flatten(dataset['X'].t()), torch.flatten(dataset['y'].t())
        sequences_num = len(x) // sequence_length
        sequences_length = int(sequences_num * sequence_length)
        x, y = x[:sequences_length], y[:sequences_length]
        if xs is None and ys is None:
            xs, ys = x, y
        else:
            xs = torch.cat([xs, x], dim=0)
            ys = torch.cat([ys, y], dim=0)

    print(f"Всего временных меток {len(xs)}")

    sequences_num = len(xs) // sequence_length
    xs = xs.reshape([sequence_length, sequences_num])
    ys = ys.reshape([sequence_length, sequences_num])

    train_num = int(sequences_num * train_part)
    val_num = int(sequences_num * val_part)

    print(f"При длине секции {sequence_length} временных меток:")
    print(f"\tЧисло сэмплов обучающей выборки: {train_num}")
    print(f"\tЧисло сэмплов валидационной выборки: {val_num}")
    print(f"\tЧисло сэмплов тестовой выборки: {sequences_num - train_num - val_num}")

    train_xs = xs[:, :train_num]
    val_xs = xs[:, train_num:train_num + val_num]
    test_xs = xs[:, train_num + val_num:]

    train_ys = ys[:, :train_num]
    val_ys = ys[:, train_num:train_num + val_num]
    test_ys = ys[:, train_num + val_num:]

    train_dataset = {'X': train_xs, 'y': train_ys}
    val_dataset = {'X': val_xs, 'y': val_ys}
    test_dataset = {'X': test_xs, 'y': test_ys}

    return train_dataset, val_dataset, test_dataset


datasets = []
for path in os.listdir('./'):
    if path.endswith('.pt'):
        datasets.append(load_xy_from_pt(path))
train_dataset, val_dataset, test_dataset = create_dataset(datasets)

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

model = SimpleNet()
model.train()
