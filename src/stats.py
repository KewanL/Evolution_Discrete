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


def plot_eda(eda):

    plt.figure()
    plt.plot(eda.history_best)
    plt.title("EDA - Best fitness")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(eda.history_mean)
    plt.title("EDA - Mean fitness")
    plt.grid()
    plt.show()


def plot_monte_carlo_eda(results):

    plt.figure()
    plt.boxplot(results)
    plt.title("Monte Carlo EDA")
    plt.grid()
    plt.show()



def plot_comparaison(ga, eda):

    # Best  
    plt.figure()
    plt.plot(ga.history_best, label='GA')
    plt.plot(eda.history_best, label='EDA')
    plt.title("Comparaison des meilleurs scores")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid()
    plt.show()

    # Mean 
    plt.figure()
    plt.plot(ga.history_mean, label='GA')
    plt.plot(eda.history_mean, label='EDA')
    plt.title("Comparaison des scores moyens")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    
    plt.legend()
    plt.grid()
    plt.show()

def plot_comparaison_monte_carlo(results_ga, results_eda):
    plt.figure()
    plt.boxplot([results_ga, results_eda], labels=['GA', 'EDA'])
    plt.title("Comparaison Monte Carlo GA vs EDA")
    plt.ylabel("Best fitness")
    plt.grid()
    plt.show()

