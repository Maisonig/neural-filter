import matplotlib.pyplot as plt

from numpy import arange


def plot_losses(losses, selection='Train'):
    plt.figure(figsize=(15, 6))
    time = arange(0, len(losses))
    plt.title("Losses")
    plt.plot(time, losses, label='Потери (loss)', alpha=1, linewidth=1)
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.savefig(f'./{selection}_losses.png')
