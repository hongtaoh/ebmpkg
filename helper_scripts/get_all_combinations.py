j_values = [50, 200, 500]
r_values = [0.1, 0.25, 0.5, 0.75, 0.9]
m_values = range(50)  # From 0 to 49 (inclusive)
# Generate all combinations
combinations = [f"{j} {r} {m}" for j in j_values for r in r_values for m in m_values]

all_combinations_file = '../all_combinations.txt'
with open(all_combinations_file, "w") as f:
    for item in combinations:
        f.write(f"{item}\n")