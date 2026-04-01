from src.utils import model_et_dictionnaire
from src.evaluator import Evaluateur
from src.Algorithme_génétique import AlgorithmeGenetique
from src.Estimation_of_Distribution_Algorithm import EstimationDistributionAlgorithm
from src.stats import plot_best, plot_mean, plot_monte_carlo, plot_eda, plot_monte_carlo_eda, plot_comparaison, plot_comparaison_monte_carlo
from src.monte_carlo import monte_carlo, monte_carlo_eda
import numpy as np

def main():

    # On charge le modèle et le dictionnaire
    model, dictionnaire = model_et_dictionnaire()
    evaluateur = Evaluateur(model, dictionnaire)

    # Premier algorithme génétique
    ga = AlgorithmeGenetique(evaluator=evaluateur, population_size=50, generations=30)
    ga.run()
    #plot_mean(ga) # Affichage

    # Monte Carlo GA
    print("Algorithme algo génétique")
    results = monte_carlo(evaluateur, 10)
    mean = np.mean(results)
    std = np.std(results)

    print(f"Monte Carlo Results: Mean = {mean:.4f}, Ecart-type = {std:.4f}")
    
    #plot_monte_carlo(results)

    # __________________________________

    # EDA
    eda = EstimationDistributionAlgorithm(evaluator=evaluateur, population_size=50, generations=30)
    eda.run()
    #plot_eda(eda)

    # Monte Carlo EDA
    print("Algorithme EDA")
    results_eda = monte_carlo_eda(evaluateur, 10)
    mean_eda = np.mean(results_eda)
    std_eda = np.std(results_eda)
    print(f"Monte Carlo EDA Results: Mean = {mean_eda:.4f}, Ecart-type = {std_eda:.4f}")
    #plot_monte_carlo_eda(results_eda)

    # Comparaison
    plot_comparaison(ga, eda)

    # Comparaison Monte Carlo
    plot_comparaison_monte_carlo(results, results_eda)

if __name__ == "__main__":
    main()