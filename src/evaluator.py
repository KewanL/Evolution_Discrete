import numpy as np
import gen_lm

class Evaluateur : 
    def __init__(self, trigram_model, dictionnaire):
        self.model = trigram_model
        self.dictionnaire = dictionnaire

        self.min_len = 4
        self.max_len = 16
        self.max_repeat = 3
        self.penalty_dict = 500.0
        self.penalty_length = 10.0
        self.penalty_repeat = 20.0
        self.alphabet = "abcdefghijklmnopqrstuvwxyz" 
        self.penalty_alphabet = 15.0 # Pénalité pour les mots contenant des caractères non alphabétiques
        self.calls = 0

    def evaluer(self, mot) :
        #Permet d'évaluer un mot à l'aide du modèle de langage.

        self.calls += 1

        # Perplexité
        perplexite = gen_lm.perplexité(mot, self.model)
        penalty = 0.0

        # Mot existant
        if mot in self.dictionnaire:
            penalty += self.penalty_dict

        # Longueur du mot
        # Pénalité si le mot est trop court ou trop long
        if len(mot) < self.min_len :
            penalty += self.penalty_length * (self.min_len - len(mot))
        elif len(mot) > self.max_len :
            penalty += self.penalty_length * (len(mot) - self.max_len)

        # Répétitions
        max_repetition = self._max_consecutive(mot)
        if max_repetition > self.max_repeat:
            penalty += self.penalty_repeat * (max_repetition - self.max_repeat)

        # Caractères non alphabétiques
        for char in mot:
            if char not in self.alphabet:
                penalty += self.penalty_alphabet
                


        return perplexite + penalty

    def _max_consecutive(self, mot) :
        #Permet de calculer le nombre maximum de caractères consécutifs identiques dans un mot.
        max_count = 1
        current_count = 1

        for i in range(1, len(mot)):
            if mot[i] == mot[i - 1]:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 1

        return max_count