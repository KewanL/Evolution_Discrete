import random 
import numpy as np 
from src.utils import mutation, crossover_partie, crossover_uniforme

class AlgorithmeGenetique:
    def __init__(self, evaluator, population_size=50, generations=50 ):
        self.evaluator = evaluator
        self.population_size = population_size
        self.generations = generations
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.history_best = []
        self.history_mean = []

    def initialiser_population(self):
        # Génère une population initiale de mots aléatoires.
        population = []
        for i in range(self.population_size):
            longueur_mot = random.randint(4, 16)  # Longueur aléatoire entre 4 et 16
            mot = ''.join(random.choices(self.alphabet, k=longueur_mot))  # Mot de longueur aléatoire
            population.append(mot)
        return population

    def meilleurs_individus(self, population, scores):
        # Selectionne les meilleurs individus de la population en fonction de leurs scores.
        sorted_population = [x for _, x in sorted(zip(scores, population))] # Trie
        moitie_population = sorted_population[: len(population)//2] # Garde la moitié des meilleurs individus
        
        return moitie_population

    # Boucle principale de l'algorithme génétique
    def run(self):
        population = self.initialiser_population()
        for generation in range(self.generations):
            scores = [self.evaluator.evaluer(mot) for mot in population]
            best_individuals = self.meilleurs_individus(population, scores)
            
            # Enregistre les meilleurs scores pour l'analyse
            best_score = min(scores)
            mean_score = np.mean(scores)
            self.history_best.append(best_score)
            self.history_mean.append(mean_score)

            print(f"Generation {generation+1}: Best Score = {best_score}, Mean Score = {mean_score}")

            # Crée la nouvelle population à partir des meilleurs individus
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(best_individuals, 2)
                enfant = crossover_uniforme(parent1, parent2)  # Crossover uniforme
                enfant_muté = mutation(enfant)  # Mutation
                new_population.append(enfant_muté)

            population = new_population

        return population