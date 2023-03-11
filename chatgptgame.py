import random

# create the board
board_size = 5
num_bombs = 5
board = [[0 for _ in range(board_size)] for _ in range(board_size)]

# place bombs randomly
bombs_placed = 0
while bombs_placed < num_bombs:
    row = random.randint(0, board_size - 1)
    col = random.randint(0, board_size - 1)
    if board[row][col] == 0:
        board[row][col] = "*"
        bombs_placed += 1

# calculate the number of bombs surrounding each cell
for row in range(board_size):
    for col in range(board_size):
        if board[row][col] == "*":
            continue
        bombs = 0
        for r in range(max(0, row-1), min(board_size, row+2)):
            for c in range(max(0, col-1), min(board_size, col+2)):
                if board[r][c] == "*":
                    bombs += 1
        board[row][col] = bombs

# print the board
for row in board:
    print(" ".join(str(cell) for cell in row))

# play the game
game_over = False
while not game_over:
    row = int(input("Enter row (0-4): "))
    col = int(input("Enter column (0-4): "))

    if board[row][col] == "*":
        print("Game over! You hit a bomb.")
        game_over = True
    else:
        print("No bomb at this location. You can safely uncover more cells.")
        board[row][col] = "X"

    # check if game has been won
    won = True
    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] != "X" and board[row][col] != "*":
                won = False
                break
        if not won:
            break

    if won:
        print("Congratulations! You have found all the bombs.")
        game_over = True

    # print updated board
    for row in board:
        print(" ".join(str(cell) for cell in row))
m
