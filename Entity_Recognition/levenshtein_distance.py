def lev(a, b):
    row, col = len(a), len(b)
    d = {} # dictionary "misused" as matrix
    for i in range(0, row + 1):
        d[(i, 0)] = i
    for j in range(0, col + 1):
        d[(0, j)] = j
    for i in range(1, row + 1):
        for j in range(1, col + 1):
            # delta is 1 iff mismatch of characters
            delta = int(a[i - 1] != b[j - 1])
            d[(i, j)] = min(d[(i - 1,     j)] + 1,
                            d[(    i, j - 1)] + 1,
                            d[(i - 1, j - 1)] + delta)
    matrix = [[0 for x in range(col+1)] for y in range(row+1)]
    for key, value in d.items():
        matrix[key[0]][key[1]] = value
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix]))

    return d[(row, col)]
a = "Peter"
b = "Pedro"
print("\nDistance between '" + a + "' and '" + b + "' is " + str(lev(a,b)) + "\n")


#___________________________________________________________________________________________________
### Adaptable Levenshtein for little errors

# Define all mismatches with lower penalty
lower_score_set = set([('o', 'u'), ('u', 'o'),
                       ('e', 'i'), ('i', 'e'),
                       ('p', 'b'), ('b', 'p'),
                       ('d', 't'), ('t', 'd'),
                       ('k', 'c'), ('c', 'k')])

def score(char_a, char_b, normal_cost, reduced_cost):
    if char_a == char_b:
        return 0
    elif (char_a, char_b) in lower_score_set:
        return reduced_cost
    return normal_cost

def lev_adapt(a, b, normal_cost, reduced_cost):
    row, col = len(a), len(b)
    d = {} # dictionary "misused" as matrix
    for i in range(0, row + 1):
        d[(i, 0)] = i
    for j in range(0, col + 1):
        d[(0, j)] = j
    for i in range(1, row + 1):
        for j in range(1, col + 1):
            delta=score(a[i - 1], b[j - 1], normal_cost, reduced_cost)
            d[(i, j)] = min(d[(i - 1,     j)] + normal_cost,
                            d[(    i, j - 1)] + normal_cost,
                            d[(i - 1, j - 1)] + delta)
    matrix = [[0 for x in range(col+1)] for y in range(row+1)]
    for key, value in d.items():
        matrix[key[0]][key[1]] = value
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix]))

    return d[(row, col)]

a = "Patric"
b = "Patrik"
print("\nDistance between '" + a + "' and '" + b + "' using normal Levenshtein " + str(lev(a,b)) + "\n")
print("\nDistance between '" + a + "' and '" + b + "' using adaptable Levenshtein " + str(lev_adapt(a,b, 1 , 0.5)) + "\n")