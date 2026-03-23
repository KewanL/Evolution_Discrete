import os 
import sys


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