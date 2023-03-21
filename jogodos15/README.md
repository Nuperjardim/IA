[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

# Jogo dos 15

## Instalação

```
Versão: Python 3.9.10
$ git clone https://github.com/Nuperjardim/IA.git
```

## Modo de utilização

```
python3 jogodos15.py <strategy> < config.txt

<strategy> values = DFS, BFS, IDFS, Greedy-misplaced, Greedy-Manhattan, "A*-misplaced", "A*-Manhattan"

config.txt deve ser um ficheiro de texto que contém 2 linhas (estado inicial e estado final)

Utilizar ("") ao chamar as estratégias "A*-misplaced" e "A*-Manhattan"
```

## Exemplo de utilização

### `Input:`

```
$ python3 jogodos15.py DFS < config.txt
```

### `Output:`

```
Este puzzle tem solução.
Procurando o caminho para a solução...
Usando: DFS
Caminho Encontrado!
['Direita', 'Baixo', 'Direita', 'Direita', 'Cima', 'Esquerda', 'Esquerda', 'Baixo', 'Baixo', 'Esquerda', 'Cima', 'Direita'] 12
Número de nós gerados: 47853
Tempo de execução: 18.618234 s
Memória usada: 32620.544
