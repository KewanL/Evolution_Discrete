import random
import numpy as np
from src.Algorithme_génétique import AlgorithmeGenetique
from src.Estimation_of_Distribution_Algorithm import EstimationDistributionAlgorithm

def monte_carlo(evaluator, iterations=1000):

    results = []
    for i in range(iterations):
        
        random.seed(i)  # Pour la reproductibilité
        np.random.seed(i)
        ga = AlgorithmeGenetique(evaluator=evaluator, population_size=50, generations=30)

        ga.run()
        results.append(ga.history_best[-1])  # Enregistre le meilleur score de la dernière génération

    return results

def monte_carlo_eda(evaluator, iterations=1000):

    results = []
    for i in range(iterations):
        
        random.seed(i)  # Pour la reproductibilité
        np.random.seed(i)
        
        eda = EstimationDistributionAlgorithm(evaluator=evaluator, population_size=50, generations=30)

        eda.run()
        results.append(eda.history_best[-1])  # Enregistre le meilleur score de la dernière génération

    return results