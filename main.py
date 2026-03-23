from src.utils import model_et_dictionnaire
from src.evaluator import Evaluateur

def main():
    model, dictionnaire = model_et_dictionnaire()
    evaluateur = Evaluateur(model, dictionnaire)
    print("Évaluation de quelques mots :")
    test = ["bonjour", "bonsoir", "bonjoure", "bonsoirrrr", "bnnjour", "allo", "salut", "ssaall"]

    for mot in test:
        score = evaluateur.evaluer(mot)
        print(f"Mot : {mot}, Score : {score}")

if __name__ == "__main__":
    main()