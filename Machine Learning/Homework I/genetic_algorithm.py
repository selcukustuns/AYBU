import numpy as np
from utils import (
    initialize_population,
    reconstruct_image,
    calculate_fitness,
    select_parents,
    crossover,
    mutate,
    elitism
)
import matplotlib.pyplot as plt

def genetic_algorithm(image_path, patch_size, population_size, generations, mutation_rate):
    population, original_image, h, w = initialize_population(image_path, patch_size, population_size)
    best_individuals = []

    for generation in range(generations):
        fitness_scores = [calculate_fitness(ind, original_image, patch_size) for ind in population]
        best_idx = np.argmax(fitness_scores)
        best_individuals.append(population[best_idx])

        print(f"Generation {generation + 1}, Best Fitness: {fitness_scores[best_idx]}")

        parents = select_parents(population, fitness_scores)
        parents = np.array(parents)  
        next_population = elitism(population, fitness_scores, num_elite=2)  

        while len(next_population) < population_size:
            if len(parents) < 2:
                raise ValueError("Parent list has fewer than 2 individuals, cannot perform crossover.")
            parent_indices = np.random.choice(len(parents), size=2, replace=False)
            parent1, parent2 = parents[parent_indices[0]], parents[parent_indices[1]]
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutate(child1, mutation_rate))
            next_population.append(mutate(child2, mutation_rate))
        population = next_population[:population_size]  

    final_best = best_individuals[-1]
    final_image = reconstruct_image(final_best, patch_size, original_image.shape)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Final Reconstructed Image")
    plt.imshow(final_image, cmap="gray")
    plt.show()
