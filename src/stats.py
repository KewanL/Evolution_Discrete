import matplotlib.pyplot as plt

def plot_best(ga):
    plt.figure()

    plt.plot(ga.history_best)
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Best fitness over generations')
    plt.legend()
    plt.grid()
    plt.show()


def plot_mean(ga):
    plt.figure()

    plt.plot(ga.history_mean)
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Mean fitness over generations')
    plt.legend()
    plt.grid()
    plt.show()

def plot_monte_carlo(results):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.boxplot(results)

    plt.title("Distribution des performances (Monte Carlo)")
    plt.ylabel("Best fitness")

    plt.grid()
    plt.show()