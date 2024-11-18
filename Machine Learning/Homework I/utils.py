import numpy as np
import cv2

def initialize_population(image_path, patch_size, population_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    patches = [image[i:i+patch_size, j:j+patch_size]
               for i in range(0, h, patch_size)
               for j in range(0, w, patch_size)]
    population = []
    for i in range(population_size):
        if i % 10 == 0:  
            population.append(patches.copy())
        else:
            np.random.shuffle(patches)
            population.append(patches.copy())
    return population, image, h, w


def reconstruct_image(individual, patch_size, image_shape):
    h, w = image_shape
    patch_per_row = w // patch_size
    reconstructed = np.zeros((h, w), dtype=np.uint8)
    for idx, patch in enumerate(individual):
        row = (idx // patch_per_row) * patch_size
        col = (idx % patch_per_row) * patch_size
        reconstructed[row:row+patch_size, col:col+patch_size] = patch
    return reconstructed


def calculate_fitness(individual, original_image, patch_size):
    reconstructed = reconstruct_image(individual, patch_size, original_image.shape)
    mse = np.mean((reconstructed - original_image) ** 2)
    return 1 / (1 + mse)  

def select_parents(population, fitness_scores):
    selected = []
    for _ in range(len(population)):
        idx1, idx2 = np.random.choice(len(population), size=2, replace=False)
        selected.append(population[idx1] if fitness_scores[idx1] > fitness_scores[idx2] else population[idx2])
    return selected

def crossover(parent1, parent2):
    child1, child2 = parent1.copy(), parent2.copy()
    for i in range(len(parent1)):
        if np.random.rand() < 0.5:
            child1[i], child2[i] = parent2[i], parent1[i]
    return child1, child2

def mutate(individual, mutation_rate=0.1):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            swap_idx = np.random.randint(0, len(individual))
            individual[i], individual[swap_idx] = individual[swap_idx], individual[i]
    return individual

def elitism(population, fitness_scores, num_elite=2):
    elite_indices = np.argsort(fitness_scores)[-num_elite:]
    return [population[i] for i in elite_indices]
