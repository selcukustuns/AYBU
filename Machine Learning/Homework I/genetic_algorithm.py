import numpy as np
from utils import (
    initialize_population,
    reconstruct_image,
    calculate_fitness,
    select_parents,
    crossover,
    mutate,
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
        next_population = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutate(child1, mutation_rate))
            next_population.append(mutate(child2, mutation_rate))
        population = next_population

    final_best = best_individuals[-1]
    final_image = reconstruct_image(final_best, patch_size, original_image.shape)

    # Visualize the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Final Reconstructed Image")
    plt.imshow(final_image, cmap="gray")
    plt.show()

    return best_individuals
