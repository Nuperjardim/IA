import sys
import math
import time
import resource
from queue import PriorityQueue

game_size = 16
n_nodes = 0


class Node:
    def __init__(self, state, parent, operator, depth, path_cost):
        self.state = state
        self.parent = parent
        self.children = list()
        self.operator = operator
        self.depth = depth
        self.path_cost = path_cost

    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def is_samePuzzle(self, state):
        if self.state == state:
            return True
        else:
            return False

    def move(self):
        new_state = self.state[:]
        blank_index = self.state.index(0)       # método para encontrar o espaço branco do puzzle
        global n_nodes

        if blank_index not in [0, 1, 2, 3]:  # CIMA
            tmp = new_state[blank_index - 4]
            new_state[blank_index - 4] = new_state[blank_index]
            new_state[blank_index] = tmp
            child = Node(new_state, self, "Cima", self.depth + 1, self.path_cost)
            self.children.append(child)
            n_nodes += 1

        new_state = self.state[:]
        if blank_index not in [12, 13, 14, 15]:  # BAIXO
            tmp = new_state[blank_index + 4]
            new_state[blank_index + 4] = new_state[blank_index]
            new_state[blank_index] = tmp
            child = Node(new_state, self, "Baixo", self.depth + 1, self.path_cost)
            self.children.append(child)
            n_nodes += 1

        new_state = self.state[:]
        if blank_index not in [0, 4, 8, 12]:  # ESQ
            tmp = new_state[blank_index - 1]
            new_state[blank_index - 1] = new_state[blank_index]
            new_state[blank_index] = tmp
            child = Node(new_state, self, "Esquerda", self.depth + 1, self.path_cost)
            self.children.append(child)
            n_nodes += 1

        new_state = self.state[:]
        if blank_index not in [3, 7, 11, 15]:  # DIR
            tmp = new_state[blank_index + 1]
            new_state[blank_index + 1] = new_state[blank_index]
            new_state[blank_index] = tmp
            child = Node(new_state, self, "Direita", self.depth + 1, self.path_cost)
            self.children.append(child)
            n_nodes += 1


def thereIsSolution(config):
    inv_count = 0   # número de inversões
    cell_row = math.ceil((16 - config.index(0)) / 4)  # linha que contém o espaço em branco
    for i in range(0, game_size):
        for j in range(i + 1, game_size):
            if config[i] > config[j] != 0:
                inv_count += 1
    return (cell_row % 2 != 0) == (inv_count % 2 == 0)


def contains(listNode, Node):
    containing = False
    for node in listNode:
        if node.is_samePuzzle(Node.state):
            containing = True
    return containing


def path_solution(Node):
    node = Node
    directions = list()
    directions.append(node.operator)
    depth = node.depth
    while node.parent is not None:
        node = node.parent
        directions.append(node.operator)
    directions.pop()
    print(directions, depth)


def DFS(initialConfig, finalConfig):   # busca em profundidade (Depth-first search)
    stack = list()
    visited = list()
    stack.append(Node(initialConfig, None, "", 0, 0))
    GoalFound = False
    max_depth = 15

    while stack and not GoalFound:
        node = stack.pop()
        visited.append(node)
        node.move()
        if max_depth > node.depth:
            for child in node.children:
                if child.is_samePuzzle(finalConfig):
                    print("Caminho Encontrado!")
                    GoalFound = True
                    path_solution(child)
                if not contains(stack, child) and not contains(visited, child):
                    stack.append(child)


def BFS(initialConfig, finalConfig):   # busca em largura (Breath-first search)
    queue = list()
    visited = list()
    queue.append(Node(initialConfig, None, "", 0, 0))
    GoalFound = False

    while queue and not GoalFound:
        node = queue.pop(0)
        visited.append(node)
        node.move()
        for child in node.children:
            if child.is_samePuzzle(finalConfig):
                print("Caminho Encontrado!")
                GoalFound = True
                path_solution(child)
            if not contains(queue, child) and not contains(visited, child):
                queue.append(child)


def DLS(initialConfig, finalConfig, depth):  # busca limitada em profundidade (Depth-limited), utilizada com a IDFS
    stack = list()
    visited = list()
    stack.append(Node(initialConfig, None, "", 0, 0))
    GoalFound = False

    while stack:
        node = stack.pop()
        visited.append(node)
        node.move()
        if depth > node.depth:
            for child in node.children:
                if child.is_samePuzzle(finalConfig):
                    print("Caminho Encontrado!")
                    GoalFound = True
                    path_solution(child)
                    return GoalFound
                if not contains(stack, child) and not contains(visited, child):
                    stack.append(child)
    return GoalFound


def IDFS(initialConfig, finalConfig):  # busca iterativa limitada em profundidade(Iterative deepening depth-first)
    depth = 0
    GoalFound = False
    while not GoalFound:
        if DLS(initialConfig, finalConfig, depth):
            GoalFound = True
        depth += 1


def heuristic_misplace(state, finalConfig):
    h = 0
    for i in range(0, 16):
        if state[i] != finalConfig[i]:
            h += 1
    return h


def heuristic_manhattan(state, finalConfig):
    cont = 0
    for i in range(0, 16):
        cont += manhattan_aux(state.index(i), finalConfig.index(i))
    return cont


def manhattan_aux(i, j):
    matrix_ij = {0: (1, 1), 0.25: (1, 2), 0.50: (1, 3), 0.75: (1, 4),
                 1: (2, 1), 1.25: (2, 2), 1.50: (2, 3), 1.75: (2, 4),
                 2: (3, 1), 2.25: (3, 2), 2.50: (3, 3), 2.75: (3, 4),
                 3: (4, 1), 3.25: (4, 2), 3.50: (4, 3), 3.75: (4, 4)}

    x1, y1 = matrix_ij[i / 4]
    x2, y2 = matrix_ij[j / 4]
    return abs(x1 - x2) + abs(y1 - y2)


def Greedy(initialConfig, finalConfig, heuristic):   # Gulosa
    visited = list()
    pq = PriorityQueue()
    pq.put(Node(initialConfig, None, "", 0, 0))
    GoalFound = False

    while pq and not GoalFound:
        node = pq.get()
        visited.append(node)
        node.move()
        for child in node.children:
            if child.is_samePuzzle(finalConfig):
                print("Caminho Encontrado!")
                GoalFound = True
                path_solution(child)
            if not contains(pq.queue, child) and not contains(visited, child):
                if heuristic == '1':                                                # heuristica misplace
                    cost = heuristic_misplace(child.state, finalConfig)
                    child.path_cost = cost
                else:                                                               # heuristica manhattan
                    cost = heuristic_manhattan(child.state, finalConfig)
                    child.path_cost = cost
                pq.put(child, cost)


def A_Star(initialConfig, finalConfig, heuristic):    # A*
    pq = PriorityQueue()
    pq.put(Node(initialConfig, None, "", 0, 0))
    GoalFound = False

    while pq and not GoalFound:
        node = pq.get()
        node.move()
        for child in node.children:
            if child.is_samePuzzle(finalConfig):
                print("Caminho Encontrado!")
                GoalFound = True
                path_solution(child)
            if heuristic == '1':                                                    # heuristica misplace
                cost = child.depth + heuristic_misplace(child.state, finalConfig)
                child.path_cost = cost
            else:                                                                   # heuristica manhattan
                cost = child.depth + heuristic_manhattan(child.state, finalConfig)
                child.path_cost = cost
            pq.put(child, cost)


def execute(strategy, initialConfig, finalConfig):
    print("Procurando o caminho para a solução...")
    start = time.time()
    global n_nodes

    if strategy == 'DFS':
        print("Usando: DFS")
        DFS(initialConfig, finalConfig)
    elif strategy == 'BFS':
        print("Usando: BFS")
        BFS(initialConfig, finalConfig)
    elif strategy == 'IDFS':
        print("Usando: IDFS")
        IDFS(initialConfig, finalConfig)
    elif strategy == 'Greedy-misplaced':
        print("Usando: Greedy-misplaced")
        Greedy(initialConfig, finalConfig, 1)
    elif strategy == 'Greedy-Manhattan':
        print("Usando Greedy-Manhattan")
        Greedy(initialConfig, finalConfig, 0)
    elif strategy == 'A*-misplaced':
        print("Usando A*-misplaced")
        A_Star(initialConfig, finalConfig, 1)
    elif strategy == 'A*-Manhattan':
        print("Usando A*-Manhattan")
        A_Star(initialConfig, finalConfig, 0)

    end = time.time()
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000
    print("Número de nós gerados: %d" % n_nodes)
    print("Tempo de execução: %f s" % (end - start))
    print("Memória usada: %s" % memory)


def main():
    strategy = sys.argv[1]
    input_lines = sys.stdin.readlines()
    initialConfig = list(map(int, input_lines[0].split()))
    finalConfig = list(map(int, input_lines[1].split()))

    if not (thereIsSolution(initialConfig) == thereIsSolution(finalConfig)):
        print("Este puzzle não tem solução.")
    else:
        print("Este puzzle tem solução.")
        execute(strategy, initialConfig, finalConfig)


if __name__ == '__main__':
    main()
