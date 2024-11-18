from genetic_algorithm import genetic_algorithm

IMAGE_PATH = "images/puzzle_image.png"
PATCH_SIZE = 8
POPULATION_SIZE = 20
GENERATIONS = 100
MUTATION_RATE = 0.1

if __name__ == "__main__":
    best_individuals = genetic_algorithm(
        IMAGE_PATH, PATCH_SIZE, POPULATION_SIZE, GENERATIONS, MUTATION_RATE
    )
