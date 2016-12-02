import random
import numpy as np
import sys

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

def fitnessFromDNA64(individual):
	board = buildBoardFromDNA64(individual)
	return fitnessFromBoard(board),

ind = [random.randint(1,9) for i in range(64)]
print(ind)
board = buildBoardFromDNA64(ind)
printBoard(board)
print(fitnessFromDNA64(ind))