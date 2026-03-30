from src.utils import model_et_dictionnaire
from src.evaluator import Evaluateur
from src.Algorithme_génétique import AlgorithmeGenetique
from src.stats import plot_best, plot_mean, plot_monte_carlo
from src.monte_carlo import monte_carlo
import numpy as np

def main():

    # On charge le modèle et le dictionnaire
    model, dictionnaire = model_et_dictionnaire()
    evaluateur = Evaluateur(model, dictionnaire)

    # Premier algorithme génétique
    # ga = AlgorithmeGenetique(evaluator=evaluateur, population_size=50, generations=30)
    # population = ga.run()
    # plot_mean(ga) # Affichage

    # Monte Carlo
    results = monte_carlo(evaluateur, 10)
    mean = np.mean(results)
    std = np.std(results)

    print(f"Monte Carlo Results: Mean = {mean:.4f}, Ecart-type = {std:.4f}")
    
    plot_monte_carlo(results)


if __name__ == "__main__":
    main()