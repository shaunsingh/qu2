import os
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

POPULATION_SIZE = 250
GENERATIONS = 25000

class GeneticStrategy:
    def __init__(self, num_towers: int, num_soldiers: int, population_size: int, generations: int):
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
        
        while np.sum(child) != self.num_soldiers:
            if np.sum(child) < self.num_soldiers:
                child[random.randint(0, self.num_towers - 1)] += 1
            else:
                non_zero_indices = np.nonzero(child)[0]
                child[random.choice(non_zero_indices)] -= 1
        return child

    def mutate(self, strategy: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        mutated = strategy.copy()
        for i in range(self.num_towers):
            if random.random() < mutation_rate:
                change = random.randint(-2, 2)
                if mutated[i] + change >= 0:
                    mutated[i] += change
        
        while np.sum(mutated) != self.num_soldiers:
            if np.sum(mutated) < self.num_soldiers:
                mutated[random.randint(0, self.num_towers - 1)] += 1
            else:
                non_zero_indices = np.nonzero(mutated)[0]
                mutated[random.choice(non_zero_indices)] -= 1
        return mutated

    def fitness(self, strategy: np.ndarray) -> float:
        score = 0
        opponent_strategy = self.random_strategy()
        for i in range(self.num_towers):
            if strategy[i] > opponent_strategy[i]:
                score += i + 1
        return score

    def evolve(self) -> Tuple[np.ndarray, List[float]]:
        population = [self.random_strategy() for _ in range(self.population_size)]
        fitness_history = []
        
        for _ in range(self.generations):
            fitness_scores = [self.fitness(strategy) for strategy in population]
            fitness_history.append(max(fitness_scores))
            
            new_population = []
            for _ in range(self.population_size):
                parent1 = random.choices(population, weights=fitness_scores)[0]
                parent2 = random.choices(population, weights=fitness_scores)[0]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
                
            population = new_population
            
        best_strategy = max(population, key=lambda x: self.fitness(x))
        return best_strategy, fitness_history
    
def plot_fitness_history(fitness_history: List[float], title: str, filename: str):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.savefig(filename)
    plt.close()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_simulation():
    num_towers = 10
    num_soldiers = 100
    genetic_strategy = GeneticStrategy(num_towers, num_soldiers, POPULATION_SIZE, GENERATIONS)

    output_dir = "genetic_algorithm_output"
    ensure_dir(output_dir)

    print("Evolving strategy against itself...")
    best_strategy, fitness_history = genetic_strategy.evolve()

    plot_fitness_history(fitness_history, 'Fitness History (Self-play)', 
                         os.path.join(output_dir, 'fitness_history_self_play.png'))

    # Display final results
    print("\nFinal Strategy (evolved through self-play):")
    print(best_strategy)

if __name__ == "__main__":
    run_simulation()