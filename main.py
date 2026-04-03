from src.utils import model_et_dictionnaire
from src.evaluator import Evaluateur
from src.Algorithme_génétique import AlgorithmeGenetique
from src.Estimation_of_Distribution_Algorithm import EstimationDistributionAlgorithm
from src.stats import plot_best, plot_mean, plot_monte_carlo, plot_eda, plot_monte_carlo_eda, plot_comparaison, plot_comparaison_monte_carlo, plot_histogramme
from src.monte_carlo import monte_carlo, monte_carlo_eda
import numpy as np
import argparse
from src.annexe import save_words_csv


def main():

    parser = argparse.ArgumentParser(description="Comparaison GA vs EDA")

    # ======================
    # PARAMÈTRES
    # ======================
    parser.add_argument("--algo", type=str, default="all",
                        choices=["ga", "eda", "all"])

    parser.add_argument("--n_runs", type=int, default=20)

    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=30)

    parser.add_argument("--mutation_rate", type=float, default=0.1)
    parser.add_argument("--crossover", type=str, default="uniforme",
                        choices=["uniforme", "partie"])

    parser.add_argument("--use_stallion", type=int, default=1)
    parser.add_argument("--use_losers", type=int, default=1)
    parser.add_argument("--use_elitism", type=int, default=1)
    parser.add_argument("--use_reseed", type=int, default=1)

    args = parser.parse_args()

    # ======================
    # INIT
    # ======================
    model, dictionary = model_et_dictionnaire()
    evaluator = Evaluateur(model, dictionary)

    # ======================
    # GA
    # ======================
    if args.algo in ["ga", "all"]:

        print("\n=== GA ===")

        results_ga, words_ga = monte_carlo(
            evaluator,
            n_runs=args.n_runs,
            population_size=args.pop_size,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_type=args.crossover,
            use_stallion=bool(args.use_stallion),
            use_losers=bool(args.use_losers),
            use_elitism=bool(args.use_elitism),
            use_reseed=bool(args.use_reseed)
        )

        print("GA Mean:", np.mean(results_ga))
        print("GA Std:", np.std(results_ga))

        plot_monte_carlo(results_ga)
        plot_histogramme(results_ga, "Distribution GA")

        save_words_csv(words_ga, evaluator, "words_ga.csv")

    # ======================
    # EDA
    # ======================
    if args.algo in ["eda", "all"]:

        print("\n=== EDA ===")

        results_eda, words_eda = monte_carlo_eda(
            evaluator,
            n_runs=args.n_runs,
            population_size=args.pop_size,
            generations=args.generations
        )

        print("EDA Mean:", np.mean(results_eda))
        print("EDA Std:", np.std(results_eda))

        plot_monte_carlo_eda(results_eda)
        plot_histogramme(results_eda, "Distribution EDA")

        save_words_csv(words_eda, evaluator, "words_eda.csv")

    # ======================
    # COMPARAISON
    # ======================
    if args.algo == "all":
        plot_comparaison_monte_carlo(results_ga, results_eda)

if __name__ == "__main__":
    main()