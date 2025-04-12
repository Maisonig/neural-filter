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
        print("Файл загружен.")
    except UnpicklingError:
        print("Не удалось загрузить данные. Неверно указан путь к файлу.")
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
                   train_num: int = 70,
                   val_num: int = 20,
                   test_num: int = 10):

    for dataset in datasets:
        pass


data = load_xy_from_pt('tensor_dataset3.pt')
a = torch.flatten(data['X'].t())
b = torch.flatten(data['y'].t())
c = {'X': a, 'y': b}
plot_series(c, sample_num=0)
