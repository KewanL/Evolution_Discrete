import random 
import numpy as np 
from src.utils import mutation, crossover_partie, crossover_uniforme

class AlgorithmeGenetique:
    def __init__(self, evaluator, population_size=50, generations=50, use_elitism = True, use_stallion = True, use_losers = True, use_reseed = True, elite_size = 2, loser_rate = 0.1, stagnation_limit = 5):
        self.evaluator = evaluator
        self.population_size = population_size
        self.generations = generations
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.history_best = []
        self.history_mean = []

        # Mécanismes de diversité
        self.use_elitism = use_elitism
        self.use_stallion = use_stallion
        self.use_losers = use_losers
        self.use_reseed = use_reseed
        self.elite_size = elite_size
        self.loser_rate = loser_rate
        self.stagnation_limit = stagnation_limit

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

    def reseed(self, population):
        # Permet de remplacer une partie de la population par de nouveaux mots aléatoires
        n = int(self.population_size * 0.2) # Ici, on dit qu'on remplace 20% de la population
        for i in range(n):
            population[-(i+1)] = ''.join(random.choices(self.alphabet, k=random.randint(4, 16)))

        return population

    


    # Boucle principale de l'algorithme génétique
    def run(self):
        population = self.initialiser_population()


        stagnation = 0
        best_global = float('inf') 


        for generation in range(self.generations):

            # évaluation de la population
            scores = [self.evaluator.evaluer(mot) for mot in population]

            # Trie 
            sorted_population = [x for _, x in sorted(zip(scores, population))]
            # best_individuals = self.meilleurs_individus(population, scores)
            
            # Enregistre les meilleurs scores pour l'analyse
            best_score = min(scores)
            mean_score = np.mean(scores)
            self.history_best.append(best_score)
            self.history_mean.append(mean_score)

            print(f"Generation {generation+1}: Best Score = {best_score}, Mean Score = {mean_score}")

            # Stagnation 
            if best_score < best_global:
                best_global = best_score
                stagnation = 0
            else : 
                stagnation += 1
            
            # Reseed 
            if self.use_reseed and stagnation >= self.stagnation_limit:
                # Dans le cas de la stagnation, 
                # on remplace une partie de la population par de nouveaux mots aléatoires pour sortir de la stagnation
                population = self.reseed(population)
                stagnation = 0
                continue

            # Crée la nouvelle population à partir des meilleurs individus
            new_population = []

            # elitisme 
            if self.use_elitism:
                new_population.extend(sorted_population[:self.elite_size]) # Garde les meilleurs individus

            # Sélection
            selection = self.meilleurs_individus(population, scores)

            # losers 
            if self.use_losers:
                nb_losers = int(self.population_size * self.loser_rate)
                losers = sorted_population[-nb_losers:] # Garde les pires individus
                selection += losers
            
            # étalon
            best_individu = sorted_population[0] # Meilleur individu de la génération


            # Reproduction
            while len(new_population) < self.population_size:

                if self.use_stallion :
                    parent1 = best_individu
                else : 
                    parent1 = random.choice(selection)
                
                parent2 = random.choice(selection)
                # parent1, parent2 = random.sample(selection, 2)
                enfant = crossover_uniforme(parent1, parent2)  # Crossover uniforme
                enfant = mutation(enfant)  # Mutation
                new_population.append(enfant)

            population = new_population

        return population