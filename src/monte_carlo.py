import random
import numpy as np
from src.Algorithme_génétique import AlgorithmeGenetique


def monte_carlo(evaluator, iterations=1000):

    results_best = []
    for i in range(iterations):
        
        random.seed(i)  # Pour la reproductibilité
        np.random.seed(i)
        ga = AlgorithmeGenetique(evaluator=evaluator, population_size=50, generations=30)

        ga.run()
        results_best.append(ga.history_best[-1])  # Enregistre le meilleur score de la dernière génération

    return results_best