import numpy as np
import sys
import matplotlib.pyplot as plt

class Sudoku():

	def __init__(self):
		self.reset()

	def reset(self):
		self.board = (np.indices((9,9)) + 1)[1]
		for i in range(len(self.board)):
			self.board[i] = np.random.permutation(self.board[i])
		self.fixedValues = np.array([
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
		self.setup()
	
	def printBoard(self, board=[]):
		if(board == []):
			board = self.board
		
		for i in range(len(board)):
			if(i % 3 == 0 and i != 0):
				print("------+------+------")
			for j in range(len(board[i])):
				if(j % 3 == 0 and j != 0):
					sys.stdout.write("|")
				sys.stdout.write(str(board[i][j]) + " ")
			print("")

	def swapToPlace(self, val, line, col):
		valIndex = np.where(self.board[line]==val)[0][0]
		self.swap(self.board[line], valIndex, col)

	def setup(self):
		for (val, row, col) in self.fixedValues:
			self.swapToPlace(val, row, col)

	def fitness(self, board=[]):
		if(board == []):
			board = self.board
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

	def swap(self, arr, pos1, pos2):
		arr[pos1], arr[pos2] = arr[pos2], arr[pos1]

	def isFixed(self, row, col):
		for t in self.fixedValues:
			if(row == t[1] and col == t[2]):
				return True
		return False

	def bestNeighbor(self):
		tempBoard = self.board.copy()
		# best = (row, (col1, col2), val)
		# col1 e col2 serão trocadas com o swap.
		best = (0, (0,0), -1)
		for i in range(len(tempBoard)):
			for j in range(len(tempBoard[i])):
				for k in range(i,len(tempBoard)):
					if(self.isFixed(i,j) or self.isFixed(i,k)):
						continue
					self.swap(tempBoard[i], j, k)
					contestant = (i, (j,k), self.fitness(tempBoard))
					if(contestant[2] > best[2]):
						best = contestant
					#Desfaz o swap para poder reutilizar o tabuleiro
					self.swap(tempBoard[i], j, k)
		return best

	def climbHill(self):
		scores = []
		maxScore = self.fitness()
		# print("Initial score: " + str(maxScore))
		while True:
			# print("Current score: " + str(maxScore))
			scores.append(maxScore)
			(row, (col1, col2), nextScore) = self.bestNeighbor()
			if(nextScore <= maxScore):
				return scores
			self.swap(self.board[row], col1, col2)
			maxScore = nextScore

sud = Sudoku()
print("Hill Climbing")
print("Quanto maior a pontuação, melhor. (Max = 243)")
print("A potuação reflete o número de valores únicos por linha, coluna e quadrante.")
trials = []
maxScore = -1
bestBoard = []
for i in range(10):
	sud.reset()
	finalScore = sud.climbHill()
	maxFinalScore = max(finalScore)
	if(maxScore < maxFinalScore):
		maxScore = maxFinalScore
		bestBoard = sud.board.copy()
	print(str(i) + ") " + str(finalScore[-1]) + "/243")
	if(finalScore == 243):
		print("SOLUÇÃO CORRETA!")
		sud.printBoard()
		break
	trials.append(finalScore)
	# print(finalScore)
print("Melhor pontuação: %i" % maxScore)
sud.printBoard(bestBoard)


#Desenha um gráfico do desempenho de cada execução do hill climbing
for trial in trials:
	plt.plot(trial)
plt.title('Hill Climbing')
plt.ylabel('Pontuação (Max 243)')
plt.xlabel('Iterações')
plt.show()