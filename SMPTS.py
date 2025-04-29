import numpy as np
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from collections import defaultdict 
from mpl_toolkits.mplot3d import Axes3D 
import os

# Create a directory for saving output plots and results
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Set random seeds for reproducibility across multiple runs
# This ensures the same random numbers are generated each time the algorithm is executed
random.seed(42)
np.random.seed(42)

# Data loading section - Parse the problem instance from external file
file_path = "data/data_4_23_59_33.dat"

with open(file_path, 'r') as file:
    data = file.readlines()

# Parse problem parameters from the data file
# Extract multi-skilling level (percentage of workers who can handle multiple jobs)
multi_skilling_level = int(next(line for line in data if "Multi-skilling level" in line).split('=')[-1].strip())
# Extract problem type identifier
type_value = int(next(line for line in data if "Type" in line).split('=')[-1].strip())
# Extract total number of jobs to be assigned
jobs_count = int(next(line for line in data if "Jobs" in line).split('=')[-1].strip())
# Parse job times as tuples of (start_time, end_time) for each job
job_time = [tuple(map(int, line.strip().split())) for line in data if line.strip().replace(' ', '').isdigit()]
# Extract number of available shifts/workers
qualifications_count = int(next(line for line in data if "Qualifications" in line).split('=')[-1].strip())
# Parse qualifications matrix - which jobs each shift is qualified to handle
qualifications = [list(map(int, line.split(':', 1)[-1].strip().split())) for line in data if ':' in line]

# Problem parameters - defining constants used throughout the algorithm
NUM_JOBS = jobs_count  # Total number of jobs to be assigned
NUM_SHIFTS = qualifications_count  # Total number of available shifts/workers
JOB_TIMES = job_time  # Time windows for each job (start and end times)
QUALIFICATIONS = qualifications  # Jobs each shift is qualified to handle
MULTI_SKILLING_LEVEL = multi_skilling_level  # Percentage of workers who can handle overlapping jobs

# Calculate maximum allowed overlaps per shift based on multi-skilling level
# Higher multi-skilling level means more overlaps can be handled by a worker
# This is a key constraint in the model representing worker capacity
MAX_ALLOWED_OVERLAPS = (MULTI_SKILLING_LEVEL / 100) * NUM_SHIFTS

print(f"Multi-skilling level: {MULTI_SKILLING_LEVEL}%")
print(f"Maximum allowed overlaps per shift: {MAX_ALLOWED_OVERLAPS:.2f}")

# Genetic Algorithm parameters - controls the evolutionary search process
POPULATION_SIZE = 20000  # Size of the population in each generation - larger populations explore more of the search space
P_CROSSOVER = 0.7        # Probability of applying crossover operation - controls exploration vs exploitation balance
P_MUTATION = 0.3         # Probability of applying mutation operation - helps maintain diversity
MAX_GENERATIONS = 150    # Maximum number of generations to evolve - limits computational time
HOF_SIZE = 10            # Number of best solutions to keep in the Hall of Fame - represents the Pareto front

# Create fitness and individual classes using DEAP's creator
# Defining a multi-objective fitness with three objectives, all to be minimized (-1.0 weight)
# The three objectives are: (1) number of shifts used, (2) maximum overlap for any shift, (3) excess overlaps beyond limit
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
# Individual is a list (representing shift assignments for each job) with the defined fitness function
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Initialize DEAP toolbox
toolbox = base.Toolbox()

# Genetic operators

# Function to create a random initial individual (solution)
def create_individual():
    individual = []
    for job in range(NUM_JOBS):
        # For each job, choose a random shift from those qualified to handle it
        qualified_shifts = [shift for shift in range(NUM_SHIFTS) if job in QUALIFICATIONS[shift]]
        if not qualified_shifts:
            raise ValueError(f"Job {job} has no qualified shifts!")
        individual.append(random.choice(qualified_shifts))
    return individual

# Evaluation function to calculate the three objectives
def evaluate(individual):
    # Objective 1: Number of distinct shifts used (minimize)
    num_shifts_used = len(set(individual))
    
    # Objective 2: Calculate maximum overlap for any shift (minimize)
    shift_overlaps = defaultdict(int)
    
    # Check all pairs of jobs for overlaps
    for i in range(NUM_JOBS):
        for j in range(i+1, NUM_JOBS):
            if individual[i] == individual[j]:  # Same shift assigned to both jobs
                start_i, end_i = JOB_TIMES[i]
                start_j, end_j = JOB_TIMES[j]
                # Check if the jobs overlap in time
                if not (end_i <= start_j or end_j <= start_i):
                    shift_overlaps[individual[i]] += 1
    
    max_overlap = max(shift_overlaps.values()) if shift_overlaps else 0
    
    # Objective 3: Penalty for exceeding the maximum allowed overlaps (minimize)
    # Calculate the total excess of overlaps beyond the allowed limit
    excess_overlaps = sum(max(0, overlaps - MAX_ALLOWED_OVERLAPS) 
                         for shift, overlaps in shift_overlaps.items())
    
    return num_shifts_used, max_overlap, excess_overlaps

# Two-point crossover operator for creating offspring
def cxTwoPoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    # Ensure cxpoint2 > cxpoint1
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    
    # Swap segments between the two crossover points
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    
    return ind1, ind2

# Mutation operator - randomly changes shift assignments while respecting qualifications
def mutUniform(individual):
    for i in range(len(individual)):
        if random.random() < 0.1:  # 10% chance to mutate each job assignment
            # Get shifts qualified to handle this job
            qualified_shifts = [shift for shift in range(NUM_SHIFTS) if i in QUALIFICATIONS[shift]]
            if qualified_shifts:
                # Randomly assign a new qualified shift
                individual[i] = random.choice(qualified_shifts)
    return individual,

# Register genetic operators with the DEAP toolbox
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", cxTwoPoint)
toolbox.register("mutate", mutUniform)
toolbox.register("select", tools.selNSGA2)  # NSGA-II selection for multi-objective optimization

# Statistics to track during evolution
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min, axis=0)  # Minimum value for each objective
stats.register("avg", np.mean, axis=0)  # Average value for each objective
stats.register("max", np.max, axis=0)  # Maximum value for each objective

# Hall of Fame to store the Pareto-optimal solutions
hof = tools.ParetoFront()

# Visualization functions

# Function to visualize job assignments using a Gantt chart
def visualize_assignment(individual, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    shift_colors = plt.cm.tab20.colors  # Color palette for shifts
    
    # Create a dictionary to track overlapping jobs per shift
    shift_jobs = defaultdict(list)
    for job_index, assigned_shift in enumerate(individual):
        shift_jobs[assigned_shift].append(job_index)
    
    # Calculate overlaps for each shift to highlight in the visualization
    shift_overlaps = defaultdict(int)
    for shift, jobs in shift_jobs.items():
        for i, job1 in enumerate(jobs):
            for job2 in jobs[i+1:]:
                start1, end1 = JOB_TIMES[job1]
                start2, end2 = JOB_TIMES[job2]
                if not (end1 <= start2 or end2 <= start1):  # Check for time overlap
                    shift_overlaps[shift] += 1
    
    # Draw horizontal bars for each job
    for job_index, assigned_shift in enumerate(individual):
        start_time, end_time = JOB_TIMES[job_index]
        
        # Use colors based on shift number, regardless of exceeding limits
        color = shift_colors[assigned_shift % len(shift_colors)]
        
        # Create the horizontal bar for this job
        ax.barh(
            y=assigned_shift,
            width=end_time - start_time,
            left=start_time,
            color=color,
            edgecolor='black',
            alpha=0.6
        )
        # Add job label to the bar
        ax.text(
            x=start_time + (end_time - start_time) / 2,
            y=assigned_shift,
            s=f"Job {job_index}",
            va='center', ha='center', color='white', fontsize=8
        )
    
    # Customize the plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Shifts")
    ax.set_title(f"Job Assignment to Shifts (Gantt Chart)\nMulti-skilling level: {MULTI_SKILLING_LEVEL}%, Max allowed overlaps: {MAX_ALLOWED_OVERLAPS:.2f}")
    ax.set_yticks(range(NUM_SHIFTS))
    ax.set_yticklabels([f"Shift {i} (Overlaps: {shift_overlaps[i]})" for i in range(NUM_SHIFTS)])
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Function to visualize shift utilization statistics
def plot_shift_utilization(individual, save_path=None):
    shifts = list(range(NUM_SHIFTS))
    
    # Calculate overlaps for each shift
    shift_overlaps = defaultdict(int)
    for i in range(NUM_JOBS):
        for j in range(i+1, NUM_JOBS):
            if individual[i] == individual[j]:  # Same shift
                start_i, end_i = JOB_TIMES[i]
                start_j, end_j = JOB_TIMES[j]
                if not (end_i <= start_j or end_j <= start_i):  # Time overlap
                    shift_overlaps[individual[i]] += 1
    
    # Count total jobs per shift
    shift_counts = defaultdict(int)
    for shift in individual:
        shift_counts[shift] += 1
    
    job_counts = [shift_counts[shift] for shift in shifts]
    overlap_counts = [shift_overlaps.get(shift, 0) for shift in shifts]

    # Create a two-panel plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top panel: Plot job counts per shift
    ax1.bar(shifts, job_counts, color="skyblue", edgecolor="black")
    ax1.set_xlabel("Shifts")
    ax1.set_ylabel("Number of Jobs")
    ax1.set_title("Shift Utilization - Total Jobs Assigned")
    ax1.set_xticks(shifts)
    ax1.set_xticklabels([f"Shift {i}" for i in shifts], rotation=45)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Bottom panel: Plot overlap counts per shift
    bars = ax2.bar(shifts, overlap_counts, color="lightgreen", edgecolor="black")
    
    # Add a horizontal line for the maximum allowed overlaps
    ax2.axhline(y=MAX_ALLOWED_OVERLAPS, color='r', linestyle='--', 
                label=f'Max Allowed Overlaps ({MAX_ALLOWED_OVERLAPS:.2f})')
    
    # Highlight bars that exceed the multi-skilling limit
    for i, count in enumerate(overlap_counts):
        if count > MAX_ALLOWED_OVERLAPS:
            bars[i].set_color('red')
    
    ax2.set_xlabel("Shifts")
    ax2.set_ylabel("Number of Overlaps")
    ax2.set_title(f"Shift Overlaps (Multi-skilling Level: {MULTI_SKILLING_LEVEL}%)")
    ax2.set_xticks(shifts)
    ax2.set_xticklabels([f"Shift {i}" for i in shifts], rotation=45)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Main algorithm function
def main():
    # Create initial population
    pop = toolbox.population(n=POPULATION_SIZE)
    
    # Evaluate the entire initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Run the genetic algorithm using DEAP's mu+lambda evolution strategy
    pop, logbook = algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=POPULATION_SIZE,          # Number of individuals to select for the next generation
        lambda_=2*POPULATION_SIZE,   # Number of children to produce at each generation
        cxpb=P_CROSSOVER,            # Probability of mating two individuals
        mutpb=P_MUTATION,            # Probability of mutating an individual
        ngen=MAX_GENERATIONS,        # Number of generations
        stats=stats,                 # Statistics object
        halloffame=hof,              # Hall of Fame to store best individuals
        verbose=True                 # Display progress information
    )
    
    return pop, logbook, hof

# Main execution block
if __name__ == "__main__":
    # Run the main algorithm
    pop, logbook, hof = main()
    
    # Print the best solutions in the Pareto front
    print("\nBest solutions in the Pareto front:")
    for i, ind in enumerate(hof):
        print(f"Solution {i+1}:")
        print(f"  Shifts used: {ind.fitness.values[0]}")
        print(f"  Max overlaps: {ind.fitness.values[1]}")
        print(f"  Excess overlaps: {ind.fitness.values[2]}")
        
        # Calculate shift-specific overlaps for detailed analysis
        shift_overlaps = defaultdict(int)
        for j in range(NUM_JOBS):
            for k in range(j+1, NUM_JOBS):
                if ind[j] == ind[k]:  # Same shift
                    start_j, end_j = JOB_TIMES[j]
                    start_k, end_k = JOB_TIMES[k]
                    if not (end_j <= start_k or end_k <= start_j):  # Time overlap
                        shift_overlaps[ind[j]] += 1
        
        # Print shifts that exceed the multi-skilling limit
        shifts_exceeding = [shift for shift, overlaps in shift_overlaps.items() 
                           if overlaps > MAX_ALLOWED_OVERLAPS]
        if shifts_exceeding:
            print(f"  Shifts exceeding multi-skilling limit ({MAX_ALLOWED_OVERLAPS:.2f}): {shifts_exceeding}")
            for shift in shifts_exceeding:
                print(f"    Shift {shift}: {shift_overlaps[shift]} overlaps")
        else:
            print(f"  All shifts are within the multi-skilling limit ({MAX_ALLOWED_OVERLAPS:.2f})")
        
        # Validation check for qualification constraints
        valid = True
        for job, shift in enumerate(ind):
            if job not in QUALIFICATIONS[shift]:
                print(f"  Invalid assignment: Job {job} assigned to unqualified shift {shift}")
                valid = False
        if valid:
            print("  All assignments are valid (qualifications constraint)")
    
    # Plot 3D Pareto front to visualize trade-offs between objectives
    front = np.array([ind.fitness.values for ind in hof])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot of solutions in objective space
    scatter = ax.scatter(
        front[:, 0],  # Number of shifts used (x-axis)
        front[:, 1],  # Maximum overlaps (y-axis)
        front[:, 2],  # Excess overlaps (z-axis)
        c=front[:, 2],  # Color by excess overlaps
        cmap='viridis',
        s=50,
        alpha=0.8
    )
    
    # Sort points for connecting with lines to visualize the Pareto front
    sorted_indices = np.argsort(front[:, 0])  # Sort by first objective
    sorted_front = front[sorted_indices]
    
    # Connect points with a line to show the Pareto front
    ax.plot(
        sorted_front[:, 0],
        sorted_front[:, 1],
        sorted_front[:, 2],
        'r-',
        linewidth=2,
        alpha=0.6,
        label='Pareto Front'
    )
    
    # Label the axes and add title
    ax.set_xlabel('Number of Shifts Used')
    ax.set_ylabel('Maximum Overlaps')
    ax.set_zlabel('Excess Overlaps')
    ax.set_title('3D Pareto Front of All Objectives')
    ax.legend()
    
    # Adjust viewing angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    pareto_path = os.path.join(output_dir, "pareto_front_3d.png")
    plt.savefig(pareto_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Create convergence plot to show how objectives improved over generations
    gen = logbook.select("gen")  # Generation numbers
    min_shifts = [entry['min'][0] for entry in logbook]  # Minimum shifts used per generation
    min_overlaps = [entry['min'][1] for entry in logbook]  # Minimum max overlaps per generation
    min_excess = [entry['min'][2] for entry in logbook]  # Minimum excess overlaps per generation

    plt.figure(figsize=(12, 8))
    plt.plot(gen, min_shifts, 'b-', marker='o', label="Minimum Shifts Used")
    plt.plot(gen, min_overlaps, 'r-', marker='s', label="Minimum Maximum Overlaps")
    plt.plot(gen, min_excess, 'g-', marker='^', label="Minimum Excess Overlaps")
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Objective Value", fontsize=12)
    plt.title("Convergence of All Objectives", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    convergence_path = os.path.join(output_dir, "convergence_plot.png")
    plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Find and visualize the best valid solution (no excess overlaps)
    best_valid_solutions = [ind for ind in hof if ind.fitness.values[2] == 0]
    if best_valid_solutions:
        print("\nBest solution with no excess overlaps:")
        # Choose the solution with minimum number of shifts among valid solutions
        best_valid = min(best_valid_solutions, key=lambda x: x.fitness.values[0])
        print(f"  Shifts used: {best_valid.fitness.values[0]}")
        print(f"  Max overlaps: {best_valid.fitness.values[1]}")
        print("  Excess overlaps: 0 (all shifts within multi-skilling limit)")
        
        # Visualize the best valid solution
        print("\nVisualizing the best valid solution...")
        gantt_path = os.path.join(output_dir, "best_valid_gantt.png")
        utilization_path = os.path.join(output_dir, "best_valid_utilization.png")
        visualize_assignment(best_valid, gantt_path)
        plot_shift_utilization(best_valid, utilization_path)
    else:
        print("\nNo solutions found with all shifts within the multi-skilling limit.")
        # If no valid solution exists, visualize the solution with minimum excess overlaps
        best_compromise = min(hof, key=lambda x: x.fitness.values[2])
        print("\nVisualizing the solution with minimum excess overlaps...")
        gantt_path = os.path.join(output_dir, "best_compromise_gantt.png")
        utilization_path = os.path.join(output_dir, "best_compromise_utilization.png")
        visualize_assignment(best_compromise, gantt_path)
        plot_shift_utilization(best_compromise, utilization_path)