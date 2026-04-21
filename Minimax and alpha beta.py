import math 

def alphabeta(d, node, ismax, val, maxd, alpha, beta):
    # If leaf node reached, return its value
    if d == maxd:
        return val[node]
    
    best = -math.inf if ismax else math.inf
    
    for i in range(2):
        value = alphabeta(d + 1, node * 2 + i, not ismax, val, maxd, alpha, beta)
        
        # Update best based on maximizing or minimizing player
        best = max(best, value) if ismax else min(best, value)
        
        # Update alpha and beta values
        if ismax:
            alpha = max(alpha, best)   # best choice for maximizer so far
        else:
            beta = min(beta, best)     # best choice for minimizer so far
        
        # Prune remaining branches if condition satisfied
        if beta <= alpha:
            break
    
    return best   # return best value after exploring/pruning


def minmax(d, node, ismax, values, maxd):
    # If leaf node reached, return its value
    if d == maxd:
        return values[node]
    
    # Recursively evaluate left and right children
    a = minmax(d + 1, node * 2, not ismax, values, maxd)
    b = minmax(d + 1, node * 2 + 1, not ismax, values, maxd)
    
    # Return max or min depending on player
    return max(a, b) if ismax else min(a, b)


if __name__ == '__main__':
    val = [3, 5, 6, 9, 1, 2, 0, -1]
    maxd = 3
    
    print("Minimax Algorithm:", minmax(0, 0, True, val, maxd))
    print("Alpha beta pruning:", alphabeta(0, 0, True, val, maxd, -math.inf, math.inf))


# Expected Output:
# Minimax Algorithm: 5
# Alpha beta pruning: 5

# Explanation:
# - Minimax explores the full game tree to find the optimal value (5).
# - Alpha-beta pruning skips unnecessary branches using alpha and beta bounds,
#   but still produces the same optimal result more efficiently.
