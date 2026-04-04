import matplotlib.pyplot as plt
import numpy as np

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

def plot_monte_carlo_ga(results):

    plt.figure()
    plt.boxplot(results)

    plt.title("Distribution des performances (Monte Carlo) - GA")
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
    plt.title("Distribution des performances (Monte Carlo) - EDA")
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


def plot_histogramme(results, title="Distribution des scores"):
    
    plt.figure()
    plt.hist(results, bins=20)
    plt.title(title)
    plt.xlabel("Fitness")
    plt.ylabel("Fréquence")
    plt.grid()
    plt.show()


def plot_boxplot_multiple(results_list, labels):

    plt.figure()
    plt.boxplot(results_list, labels=labels)
    plt.title("Comparaison des distributions")
    plt.ylabel("Fitness")
    plt.grid()
    plt.show()

def plot_convergence_mean(histories, title="Convergence moyenne"):

    mean_curve = np.mean(histories, axis=0)

    plt.figure()
    plt.plot(mean_curve)
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid()
    plt.show()

def plot_convergence_statistique(histories, title="Profil de convergence (Monte-Carlo)"):
    histories = np.array(histories)
    # Calcul des statistiques par génération
    median_curve = np.median(histories, axis=0)
    q1 = np.percentile(histories, 25, axis=0)
    q3 = np.percentile(histories, 75, axis=0)
    plt.figure(figsize=(10, 6))
    plt.fill_between(range(len(median_curve)), q1, q3, color='blue', alpha=0.2, label='Dispersion (Q1-Q3)')
    plt.plot(median_curve, color='blue', lw=2, label='Médiane')
    plt.title(title)
    plt.xlabel("Générations (Appels approx. à la fct objective)")
    plt.ylabel("Fitness (Perplexité + Pénalités)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def plot_comparaison_convergence(hist_ga, hist_eda):
    plt.figure(figsize=(10, 6))
    # Stats GA
    m_ga = np.median(hist_ga, axis=0)
    plt.plot(m_ga, label='GA (Médiane)', color='green')
    plt.fill_between(range(len(m_ga)), np.percentile(hist_ga, 25, axis=0), np.percentile(hist_ga, 75, axis=0), color='green', alpha=0.15)
    
    # Stats EDA
    m_eda = np.median(hist_eda, axis=0)
    plt.plot(m_eda, label='EDA (Médiane)', color='orange')
    plt.fill_between(range(len(m_eda)), np.percentile(hist_eda, 25, axis=0), np.percentile(hist_eda, 75, axis=0), color='orange', alpha=0.15)

    plt.title("Comparaison des Profils de Convergence : GA vs EDA")
    plt.xlabel("Générations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()