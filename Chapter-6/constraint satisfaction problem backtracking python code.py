def is_safe(board, row, col, n):
    # Check if it's safe to place a queen at board[row][col]

    # Check the row on the left side
    for i in range(col):
        if board[row][i] == 1:
            return False

    # Check upper diagonal on the left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check lower diagonal on the left side
    for i, j in zip(range(row, n, 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    return True

def solve_n_queens_util(board, col, n):
    # Base case: If all queens are placed, return True
    if col >= n:
        return True

    # Try placing a queen in each row of the current column
    for i in range(n):
        if is_safe(board, i, col, n):
            # Place the queen
            board[i][col] = 1

            # Recur to place queens in the next columns
            if solve_n_queens_util(board, col + 1, n):
                return True

            # If placing a queen in the current position doesn't lead to a solution, backtrack
            board[i][col] = 0

    # If no row is found where the queen can be placed, return False
    return False

def solve_n_queens(n):
    # Create an empty chessboard
    board = [[0 for _ in range(n)] for _ in range(n)]

    # Call the utility function to solve the N-Queens problem
    if not solve_n_queens_util(board, 0, n):
        print("Solution does not exist.")
        return

    # Print the solution
    print_solution(board)

def print_solution(board):
    # Print the N-Queens solution
    for row in board:
        print(" ".join(["Q" if cell == 1 else "." for cell in row]))

# Example: Solve the 8-Queens problem
solve_n_queens(8)
