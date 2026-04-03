import csv

def save_words_csv(words, evaluator, filename="words.csv"):

    unique_words = list(set(words))

    scored = []
    for w in unique_words:
        score = evaluator.evaluer(w)
        scored.append((w, score))

    scored.sort(key=lambda x: x[1]) # On va récupérer les meilleurs mots

    top_500 = scored[:500]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mot", "score"])
        writer.writerows(top_500)