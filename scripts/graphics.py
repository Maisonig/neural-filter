import matplotlib.pyplot as plt

from numpy import arange


def plot_losses(losses, selection='train'):
    plt.figure(figsize=(15, 6))
    time = arange(0, len(losses))
    plt.title("Losses")
    plt.plot(time, losses, label='Потери (loss)', alpha=1, linewidth=1)
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.savefig(f'../graphics/{selection}_losses.png')

def plot_metrics(metrics: dict):
    plt.figure(figsize=(15, 6))
    keys = list(metrics.keys())
    time = arange(0, len(metrics[keys[0]]))
    plt.title("Metrics")
    for key in keys:
        plt.plot(time, metrics[key], label=f'{key}', alpha=1, linewidth=1)
    plt.legend(loc='upper right', prop={'size': 14})
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('.', fontsize=14)
    plt.savefig(f'../graphics/metrics.png')
