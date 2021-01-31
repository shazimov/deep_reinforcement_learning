import matplotlib.pyplot as plt

def plot_scores(scores, title, y_label, x_label, fig_size=(25,5), color='royalblue'):
    """ Plots the scores """

    plt.figure(figsize=fig_size)
    plt.plot(scores, color=color)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()
