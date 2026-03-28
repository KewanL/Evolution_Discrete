import os 
import sys
import random


# chemin vers le code de démarage 
SARTER_CODE_PATH = os.path.abspath("external/starter_code")

if SARTER_CODE_PATH not in sys.path:
    sys.path.append(SARTER_CODE_PATH)

def model_et_dictionnaire():
    """
    Permet de charger le dictionnaire et de construire le modèle
    """

    import gen_lm
    import generate_corpus

    dictionnaire = generate_corpus.generate_dictionary()
    dictionnaire_set = set(dictionnaire)
    corpus = dictionnaire
    trigram_model = gen_lm.build_trigram_model(corpus)
    return trigram_model, dictionnaire_set




def mutation(mot, alphabet="abcdefghijklmnopqrstuvwxyz", mutation_rate=0.1):
    # Change des lettres du mot avec une certaine probabilité.
    mot_muté = list(mot)
    for i in range(len(mot_muté)):
        if random.random() < mutation_rate:
            mot_muté[i] = random.choice(alphabet)
    return ''.join(mot_muté)


def crossover_partie(mot1, mot2):
    # Combine deux mots en prenant une partie de chacun.
    if len(mot1) < 2 or len(mot2) < 2:
        return mot1  # Pas de crossover possible, retourne le mot original
    point_crossover = random.randint(1, min(len(mot1), len(mot2)) - 1)
    enfant = mot1[:point_crossover] + mot2[point_crossover:]
    return enfant

def crossover_uniforme(mot1, mot2):
    # Combine deux mots en choisissant aléatoirement chaque caractère de l'un ou de l'autre.
    enfant = []
    for c1, c2 in zip(mot1, mot2):
        enfant.append(random.choice([c1, c2]))
    return ''.join(enfant)