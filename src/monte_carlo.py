import random
import numpy as np
from src.Algorithme_génétique import AlgorithmeGenetique
from src.Estimation_of_Distribution_Algorithm import EstimationDistributionAlgorithm

def monte_carlo_ga(evaluator, n_runs = 10, population_size=50, generations=30, mutation_rate=0.1, crossover_type="uniform", use_stallion = True, use_losers = True, use_elitism = True, use_reseed = True):

    results = []
    all_words = []
    all_histories = []

    for i in range(n_runs):
        
        random.seed(i)  # Pour la reproductibilité
        np.random.seed(i)
        ga = AlgorithmeGenetique(evaluator=evaluator, population_size=population_size, generations=generations, mutation_rate=mutation_rate,  use_stallion=use_stallion, use_losers=use_losers, use_elitism=use_elitism, use_reseed=use_reseed, crossover_method=crossover_type,)

        # On recupère les mots de la meilleure solution pour les afficher à la fin
        population = ga.run()
        all_words.extend(population)  # On ajoute les mots de la population à la liste globale
        results.append(min([evaluator.evaluer(mot) for mot in population]))  # Enregistre le meilleur score de la dernière génération
        all_histories.append(ga.history_best)  # Enregistre l'historique de la meilleure solution à chaque génération

    print(" Monte Carlo - Algo GA ______________________")
    for i in range(min(10, len(all_words))):
        print(f"Run {i+1}: Best word = {all_words[i]}, Score = {results[i]}")
    return np.array(results), all_words, np.array(all_histories)

def monte_carlo_eda(evaluator, n_runs=10, population_size=50, generations=30):

    results = []
    all_words = []
    all_histories = []
    for i in range(n_runs):
        
        random.seed(i)  # Pour la reproductibilité
        np.random.seed(i)
        
        eda = EstimationDistributionAlgorithm(evaluator=evaluator, population_size=population_size, generations=generations)

        population = eda.run()
        scores = [evaluator.evaluer(mot) for mot in population]
        best_score = min(scores)
        all_words.extend(population)
        results.append(best_score)  # Enregistre le meilleur score de la dernière génération
        all_histories.append(eda.history_best)  # Enregistre l'historique de la meilleure solution à chaque génération

    print(" Monte Carlo - Algo EDA ______________________")
    for i in range(min(10, len(all_words))):
        print(f"Run {i+1}: Best word = {all_words[i]}, Score = {results[i]}")
    return np.array(results), all_words, np.array(all_histories)