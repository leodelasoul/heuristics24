def parse_input(filename):
    with open(filename, "r") as f:
        lines = f.read().strip().split("\n")

    # Parse first line
    U_size, V_size, C_size, E_size = map(int, lines[0].split())

    # Parse constraints
    constraints_start = lines.index("#constraints") + 1
    edges_start = lines.index("#edges")
    constraints = [tuple(map(int, line.split())) for line in lines[constraints_start:edges_start]]

    # Parse edges
    edges = {}
    for line in lines[edges_start + 1:]:
        u, v, weight = map(int, line.split())
        edges[(u, v)] = weight

    U = list(range(1, U_size + 1))
    V = list(range(U_size + 1, U_size + V_size + 1))

    return U, V, constraints, edges