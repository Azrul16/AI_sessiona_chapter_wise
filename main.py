from itertools import product

# Define the variables and their possible values (True, False)
variables = list(product([True, False], repeat=2))

# Function to print truth table for a given proposition
def print_truth_table(proposition, variables, expression_func):
    print(f"Truth Table for: {proposition}")
    print(f"{'A':^5} {'B':^5} {'Result':^10}")
    print("-" * 20)
    for A, B in variables:
        result = expression_func(A, B)
        print(f"{A:^5} {B:^5} {result:^10}")
    print("\n")

# Define each proposition's logical expression
# a) R if and only if C
print_truth_table("a) R if and only if C", variables, lambda R, C: R == C)

# b) If G, then A
print_truth_table("b) If G, then A", variables, lambda G, A: not G or A)

# c) A or T
print_truth_table("c) A or T", variables, lambda A, T: A or T)

# d) S or B
print_truth_table("d) S or B", variables, lambda S, B: S or B)

# e) R and I
print_truth_table("e) R and I", variables, lambda R, I: R and I)
