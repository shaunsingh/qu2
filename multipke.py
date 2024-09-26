import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from tqdm import tqdm
import multiprocessing
import os

POPULATION_SIZE = 250
GENERATIONS = 25000

class GeneticStrategy:
    def __init__(
        self, num_towers: int, num_soldiers: int, population_size: int, generations: int
    ):
        self.num_towers = num_towers
        self.num_soldiers = num_soldiers
        self.population_size = population_size
        self.generations = generations

    def random_strategy(self) -> np.ndarray:
        strategy = np.zeros(self.num_towers, dtype=int)
        soldiers_left = self.num_soldiers
        for i in range(self.num_towers - 1):
            soldiers = random.randint(0, soldiers_left)
            strategy[i] = soldiers
            soldiers_left -= soldiers
        strategy[-1] = soldiers_left
        return strategy

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        child = np.zeros(self.num_towers, dtype=int)
        split = random.randint(1, self.num_towers - 1)
        child[:split] = parent1[:split]
        child[split:] = parent2[split:]

        # Adjust to ensure the total number of soldiers is correct
        while np.sum(child) != self.num_soldiers:
            if np.sum(child) < self.num_soldiers:
                child[random.randint(0, self.num_towers - 1)] += 1
            else:
                non_zero_indices = np.nonzero(child)[0]
                child[random.choice(non_zero_indices)] -= 1

        return child

    def mutate(self, strategy: np.ndarray, mutation_rate: float = 0.2) -> np.ndarray:
        mutated = strategy.copy()
        for i in range(self.num_towers):
            if random.random() < mutation_rate:
                change = random.randint(-2, 2)
                if mutated[i] + change >= 0:
                    mutated[i] += change

        # Adjust to ensure the total number of soldiers is correct
        while np.sum(mutated) != self.num_soldiers:
            if np.sum(mutated) < self.num_soldiers:
                mutated[random.randint(0, self.num_towers - 1)] += 1
            else:
                non_zero_indices = np.nonzero(mutated)[0]
                mutated[random.choice(non_zero_indices)] -= 1

        return mutated

    def fitness(self, strategy: np.ndarray, opponent: np.ndarray) -> float:
        score = 0
        for i in range(self.num_towers):
            if strategy[i] > opponent[i]:
                score += i + 1
        return score

    def evolve(self) -> Tuple[np.ndarray, List[float]]:
        population = [self.random_strategy() for _ in range(self.population_size)]
        fitness_history = []

        for _ in tqdm(range(self.generations), desc="Evolving"):
            fitness_scores = [
                self.fitness(strategy, self.random_strategy())
                for strategy in population
            ]
            fitness_history.append(max(fitness_scores))

            new_population = []
            for _ in range(self.population_size):
                parent1 = random.choices(population, weights=fitness_scores)[0]
                parent2 = random.choices(population, weights=fitness_scores)[0]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        best_strategy = max(
            population, key=lambda x: self.fitness(x, self.random_strategy())
        )
        return best_strategy, fitness_history

    def get_mixed_strategy(self, num_variations: int) -> List[np.ndarray]:
        return [self.random_strategy() for _ in range(num_variations)]


def proportional_strategy(num_towers: int, num_soldiers: int) -> np.ndarray:
    strategy = np.array(range(1, num_towers + 1))
    return np.round(strategy / np.sum(strategy) * num_soldiers).astype(int)


def square_root_strategy(num_towers: int, num_soldiers: int) -> np.ndarray:
    strategy = np.sqrt(np.array(range(1, num_towers + 1)))
    return np.round(strategy / np.sum(strategy) * num_soldiers).astype(int)


def exponential_strategy(
    num_towers: int, num_soldiers: int, base: float = 1.5
) -> np.ndarray:
    strategy = np.power(base, np.array(range(1, num_towers + 1)))
    return np.round(strategy / np.sum(strategy) * num_soldiers).astype(int)


def sigmoid_strategy(num_towers: int, num_soldiers: int, k: float = 0.5) -> np.ndarray:
    x = np.array(range(1, num_towers + 1))
    strategy = 1 / (1 + np.exp(-k * (x - num_towers / 2)))
    return np.round(strategy / np.sum(strategy) * num_soldiers).astype(int)


def human_like_strategy(num_towers: int, num_soldiers: int) -> np.ndarray:
    # Allocate more soldiers to higher-value towers, with some randomness
    base_strategy = np.array(range(1, num_towers + 1)) ** 1.5
    noise = np.random.normal(0, 0.2, num_towers)
    strategy = base_strategy + noise
    strategy = np.maximum(strategy, 0)  # Ensure no negative values
    return np.round(strategy / np.sum(strategy) * num_soldiers).astype(int)


def adaptive_strategy(num_towers: int, num_soldiers: int) -> np.ndarray:
    # Adapt allocation based on tower value and remaining soldiers
    strategy = np.zeros(num_towers, dtype=int)
    remaining_soldiers = num_soldiers

    for i in range(num_towers - 1, -1, -1):
        allocation = min(
            int(remaining_soldiers * (i + 1) / sum(range(1, num_towers + 1))),
            remaining_soldiers,
        )
        strategy[i] = allocation
        remaining_soldiers -= allocation

    # Distribute any remaining soldiers
    while remaining_soldiers > 0:
        for i in range(num_towers - 1, -1, -1):
            if remaining_soldiers > 0:
                strategy[i] += 1
                remaining_soldiers -= 1
            else:
                break

    return strategy


def benchmark_game(args):
    strategy, opponent_strategy, num_towers, num_soldiers, _ = args
    if isinstance(opponent_strategy, GeneticStrategy):
        opponent = opponent_strategy.random_strategy()
    else:
        opponent = opponent_strategy(num_towers, num_soldiers)
    return GeneticStrategy(num_towers, num_soldiers, 100, 1000).fitness(
        strategy, opponent
    )


def benchmark(
    strategy: np.ndarray,
    opponent_strategies: List[Callable],
    weights: List[float],
    num_games: int = 1000,
) -> List[float]:
    results = []
    num_towers, num_soldiers = len(strategy), sum(strategy)

    with tqdm(total=len(opponent_strategies), desc="Benchmarking") as pbar:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for opponent_strategy, weight in zip(opponent_strategies, weights):
                if isinstance(opponent_strategy, GeneticStrategy):
                    args = [
                        (strategy, opponent_strategy, num_towers, num_soldiers, i)
                        for i in range(num_games)
                    ]
                else:
                    args = [
                        (strategy, opponent_strategy, num_towers, num_soldiers, i)
                        for i in range(num_games)
                    ]
                game_scores = pool.map(benchmark_game, args)
                weighted_score = (sum(game_scores) / num_games) * weight
                results.append(weighted_score)
                pbar.update(1)

    return results


def plot_fitness_history(fitness_history: List[float], title: str, filename: str):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig(filename)
    plt.close()


def plot_benchmark_results(
    results: List[float], opponent_names: List[str], filename: str
):
    plt.figure(figsize=(10, 6))
    plt.bar(opponent_names, results)
    plt.title("Benchmark Results")
    plt.xlabel("Opponent Strategy")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_final_strategy(strategy: np.ndarray, filename: str):
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(strategy) + 1), strategy)
    plt.title("Final Strategy")
    plt.xlabel("Tower")
    plt.ylabel("Number of Soldiers")
    plt.savefig(filename)
    plt.close()


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_simulation():
    num_towers = 10
    num_soldiers = 100
    genetic_strategy = GeneticStrategy(
        num_towers,
        num_soldiers,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
    )

    output_dir = "genetic_algorithm_output"
    ensure_dir(output_dir)

    print("Evolving strategy against itself...")
    best_strategy, fitness_history = genetic_strategy.evolve()

    # Plot and save fitness history
    plot_fitness_history(
        fitness_history,
        "Fitness History (Self-play)",
        os.path.join(output_dir, "fitness_history_self_play.png"),
    )

    # Benchmark and plot results
    print("Benchmarking best strategy...")
    opponent_strategies = [
        genetic_strategy,
        proportional_strategy,
        square_root_strategy,
        exponential_strategy,
        sigmoid_strategy,
        human_like_strategy,
        adaptive_strategy,
    ]
    opponent_names = [
        "Genetic",
        "Proportional",
        "Square Root",
        "Exponential",
        "Sigmoid",
        "Human-like",
        "Adaptive",
    ]
    # weights = [0.5, 0.5, 1.0, 1.5, 1.5, 2.0, 2.0]  # Adjust these weights as needed
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Adjust these weights as needed
    # weights = [.3, .5, .7, .9, 1.2, .2, 2.7, 3.1, .2, .2] # round1 weights
    results = benchmark(best_strategy, opponent_strategies, weights)
    plot_benchmark_results(
        results,
        opponent_names,
        os.path.join(output_dir, "benchmark_results_self_play.png"),
    )

    # Plot and save final strategy
    plot_final_strategy(
        best_strategy, os.path.join(output_dir, "final_strategy_self_play.png")
    )

    print("\nFinal Strategy (evolved through self-play):")
    print(best_strategy)

if __name__ == "__main__":
    multiprocessing.freeze_support()  # This line is necessary for Windows compatibility
    run_simulation()
