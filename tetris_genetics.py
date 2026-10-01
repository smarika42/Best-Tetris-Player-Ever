import random
from tetris_trainer import play_headless_game
import csv

# Training hyper-parameters
POPULATION_SIZE = 20
GENERATIONS = 10

def generate_random_dna():
    """Spawns a bot with completely randomized, extreme brain weights."""
    return {
        'lines_1': random.uniform(-500, 50),   # ALLOWS NEGATIVE PUNISHMENT!
        'lines_2': random.uniform(-200, 100),  # ALLOWS NEGATIVE PUNISHMENT!
        'lines_3': random.uniform(0, 500),
        'lines_4': random.uniform(1000, 5000), # MASSIVE TETRIS REWARD
        'height': random.uniform(0.1, 10),     # Force it to care about height
        'holes': random.uniform(5, 30),        # Force it to fear holes
        'bumpiness': random.uniform(0, 10),
        'wells': random.uniform(0.1, 20)       # Let it discover the well strategy
    }

def breed(parent1, parent2):
    """Mixes the DNA of two winning parents and applies random mutations."""
    child_dna = {}
    
    # 1. Crossover: Flip a coin for every trait to pick which parent it comes from
    for trait in parent1.keys():
        if random.random() > 0.5:
            child_dna[trait] = parent1[trait]
        else:
            child_dna[trait] = parent2[trait]
            
    # 2. Mutation: 10% chance to slightly alter a trait to discover new strategies
    for trait in child_dna.keys():
        if random.random() < 0.30: 
            child_dna[trait] *= random.uniform(0.5, 1.5) # Mutate by +/- 50%
            
    return child_dna

def run_evolution():
    print("🧬 Initializing Population...")
    population = [generate_random_dna() for _ in range(POPULATION_SIZE)]

    with open('training_history.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Generation', 'Fitness Score', 'Lines 1', 'Lines 2', 'Lines 3', 'Lines 4', 'Height Penalty', 'Holes Penalty', 'Bumpiness Penalty', 'Wells Reward'])

    for gen in range(GENERATIONS):
        print(f"\n--- GENERATION {gen + 1} ---")
        
        # 1. Evaluate Fitness (The Hunger Games)
        scored_population = []
        for i, dna in enumerate(population):
            score = play_headless_game(dna, piece_limit=1000) 
            scored_population.append((score, dna))
            
        # 2. Sort by highest score first
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
        best_score = scored_population[0][0]
        best_dna = scored_population[0][1]
        print(f"🏆 Generation Champion Score: {best_score}")

        # --- NEW LOGGING BLOCK ADDED HERE ---
        with open('training_history.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                gen + 1,
                best_score,
                round(best_dna['lines_1'], 2),
                round(best_dna['lines_2'], 2),
                round(best_dna['lines_3'], 2),
                round(best_dna['lines_4'], 2),
                round(best_dna['height'], 2),
                round(best_dna['holes'], 2),
                round(best_dna['bumpiness'], 2),
                round(best_dna['wells'], 2)
            ])
        # ------------------------------------

        # 3. Selection (Keep the top 3 bots to breed)
        top_performers = [bot[1] for bot in scored_population[:3]]

        # 4. Breed the next generation
        next_generation = []
        next_generation.append(best_dna) # Elitism: The champion always survives!
        
        while len(next_generation) < POPULATION_SIZE:
            p1 = random.choice(top_performers)
            p2 = random.choice(top_performers)
            child = breed(p1, p2)
            next_generation.append(child)

        population = next_generation

    print("\n🎉 EVOLUTION COMPLETE! Here is the ultimate bot DNA:")
    for trait, value in best_dna.items():
        print(f"'{trait}': {round(value, 3)},")

if __name__ == "__main__":
    run_evolution()