import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from deap import creator, base, tools, algorithms

fixedValues = np.array([
						#(val, row, col)
						(7, 0, 3),
						(1, 1, 0),
						(4, 2, 3),
						(3, 2, 4),
						(2, 2, 6),
						(6, 3, 8),
						(5, 4, 3),
						(9, 4, 5),
						(4, 5, 6),
						(1, 5, 7),
						(8, 5, 8),
						(8, 6, 4),
						(1, 6, 5),
						(2, 7, 2),
						(5, 7, 7),
						(4, 8, 1),
						(3, 8, 6)
						])

def printBoard(board):
	for i in range(len(board)):
		if(i % 3 == 0 and i != 0):
			print("------+------+------")
		for j in range(len(board[i])):
			if(j % 3 == 0 and j != 0):
				sys.stdout.write("|")
			sys.stdout.write(str(board[i][j]) + " ")
		print("")

def printBoardFromDNA64(individual):
	board = buildBoardFromDNA64(individual)
	printBoard(board)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("pos_val", random.randint, 1, 9)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.pos_val, n=64)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def fitnessFromDNA64(individual):
	board = buildBoardFromDNA64(individual)
	return fitnessFromBoard(board),

def buildBoardFromDNA64(individual):
	flattenedIdx = list(map(lambda t: t[1]*9 + t[2], fixedValues))
	values = fixedValues.T[0]
	flatBoard = []
	fixedValuesCounter = 0
	for i in range(81):
		if(i in flattenedIdx):
			flatBoard.append(values[fixedValuesCounter])
			fixedValuesCounter += 1
			continue
		flatBoard.append(individual[i - fixedValuesCounter])
	return np.array(flatBoard).reshape(9,9)


def fitnessFromBoard(board):
	score = 0
	rows, cols = board.shape
	for row in board:
		score += len(np.unique(row))
	for col in board.T:
		score += len(np.unique(col))
	for i in range(0, 3):
	    for j in range(0, 3):
	        sub = board[3*i:3*i+3, 3*j:3*j+3]
	        score += len(np.unique(sub))
	return score

toolbox.register("evaluate", fitnessFromDNA64)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=9, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("select", tools.selBest, k = 300)

population = toolbox.population(n=50)

# stats = tools.Statistics(lambda ind: ind.fitness.values)
# stats.register("avg", np.mean)
# stats.register("std", np.std)
# stats.register("min", np.min)
# stats.register("max", np.max)
gensMin = []
gensMax = []
gensAvg = []
gensStd = []

NGEN=100
for gen in range(NGEN):
	print("---GEN %i ---" % gen)
	offspring = algorithms.varAnd(population, toolbox, cxpb=0.8, mutpb=0.2)
	fits = toolbox.map(toolbox.evaluate, offspring)
	for fit, ind in zip(fits, offspring):
		ind.fitness.values = fit
	
	# Gather all the fitnesses in one list and print the stats
	fits = [ind.fitness.values[0] for ind in offspring]
	# print(list(fits))

	length = len(population)
	mean = sum(fits) / length
	sum2 = sum(x*x for x in fits)
	std = abs(sum2 / length - mean**2)**0.5

	gensMin.append(min(fits))
	gensMax.append(max(fits))
	gensAvg.append(mean)
	gensStd.append(std)

	# print("  Min %s" % int(min(fits)))
	print("  Max %s" % int(max(fits)))
	# print("  Avg %s" % mean)
	# print("  Std %s" % std)
	population = toolbox.select(offspring, k=len(population))
topk = tools.selBest(population, k=1)
#print(top10)
for solution in topk:
	print("Pontos: %i/243" % int(fitnessFromDNA64(solution)[0]))
	printBoardFromDNA64(solution)
	print("")

plt.subplot(111)
plt.plot(gensMax, label="Max")
plt.plot(gensAvg, label="Avg")
plt.plot(gensMin, label="Min")
plt.legend(bbox_to_anchor=(0.8, 0.0, 0.2, .102), loc=3, ncol=1, mode="expand", borderaxespad=0.)
plt.title('Genetic Algorithm (pi = 50, ng = 100, pc = 80%, pm = 20%)')
plt.ylabel('Pontuação (Max 243)')
plt.xlabel('Iterações')
plt.show()