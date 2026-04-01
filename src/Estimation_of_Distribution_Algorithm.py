import numpy as np
import random as random
from src.utils import model_et_dictionnaire

class EstimationDistributionAlgorithm:

    def __init__(self, evaluator, population_size=50, generations=50, longueur_mot = 16, selection_rate = 0.5):

        self.evaluator = evaluator
        self.population_size = population_size
        self.generations = generations
        self.alphabet = list("abcdefghijklmnopqrstuvwxyz#") # On ajoute un caractère de padding pour les mots plus courts
        self.history_best = []
        self.history_mean = []
        self.longueur_mot = longueur_mot
        self.selection_rate = selection_rate

    def initialiser_population(self):
        # Génère une population initiale de mots aléatoires.
        population = []
        for i in range(self.population_size):
            longueur_mot = random.randint(4, 16)  # Longueur aléatoire entre 4 et 16
            mot = ''.join(random.choices(self.alphabet, k=longueur_mot))  # Mot de longueur aléatoire
            population.append(mot)
        return population

    def calcul_proba(self, population):
        proba = np.zeros((self.longueur_mot, len(self.alphabet)))

        for mot in population : 
            mot = mot + "#" * (self.longueur_mot - len(mot))  # On applique un padding pour que les mots aient la même longueur

            for i in range(self.longueur_mot): 
                lettre = mot[i]
                for j in range(len(self.alphabet)):
                    
                    if self.alphabet[j] == lettre: # Si la lettre correspond à celle de l'alphabet, on incrémente la probabilité
                        proba[i][j] += 1
                        break
            
        
        # Normalisation des probabilités
        for i in range(self.longueur_mot):
            total = 0 
            for j in range(len(self.alphabet)):
                total += proba[i][j]

            if total > 0 : 
                for g in range(len(self.alphabet)):
                    proba[i][g] /= total
        return proba


    def generer_nouveaux_mots(self, proba):
        # Permet de générer de nouveaux mots en utilisant les probabilités calculées à partir de la population sélectionnée.
        new_mot = "" 

        for i in range(self.longueur_mot):
            lettre = np.random.choice(self.alphabet, p=proba[i])  # Choix d'une lettre en fonction des probabilités
            new_mot += lettre
        
        new_mot = new_mot.replace("#", "")  # On enlève les espaces de padding
        return new_mot
    
    def run(self): 

        population = self.initialiser_population() # Initialisation de la population

        for generation in range(self.generations):
            scores = [] # Liste pour stocker les scores 

            for mot in population : 
                #Évaluation
                score = self.evaluator.evaluer(mot) # On évalue le mot
                scores.append(score) # On ajoute le score à la liste des scores

            best_score = min(scores) # On trouve le meilleur score
            mean_score = np.mean(scores) # On calcule la moyenne des scores

            self.history_best.append(best_score) # On ajoute le meilleur score à l'historique
            self.history_mean.append(mean_score) # On ajoute la moyenne des scores à l'historique

            print(f"Generation {generation+1} : Best = {best_score:.2f}, Mean = {mean_score:.2f}")

            # Tri :
            sorted_population = [x for _, x in sorted(zip(scores, population))] 

            nb_selection = int(self.population_size * self.selection_rate) # Nombre d'individus à sélectionner
            selection = sorted_population[:nb_selection] # Sélection des meilleurs individus
            proba = self.calcul_proba(selection) # Calcul des probabilités à partir de la sélection

            # Génération de nouveaux mots
            new_population = []
            for i in range(self.population_size):
                new_mot = self.generer_nouveaux_mots(proba) # Génération d'un nouveau mot
                new_population.append(new_mot) # Ajout du nouveau mot à la nouvelle population
            population = new_population # Remplacement de l'ancienne population par la nouvelle population
        return population

