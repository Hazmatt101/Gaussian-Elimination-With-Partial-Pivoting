'''
Python Implementation of Gaussian Elimination with Partial Pivoting
Created on May 1, 2019
Credit to mathonline.wikidot.com for providing the math-focused conceptual algorithm used to make this, as well as the Worcester Polytechnic Institute for providing a more code-oriented algorithm.
@author: Matt McCortney
'''
# Required conditions: aMatrix must be size n x n and bMatrix must be size n x 1
def solveGaussian(aMatrix: [[float]], bMatrix: [float]) -> [float]:
    # Check to see that matrices are of appropriate size.
    n = len(aMatrix)
    if len(aMatrix) == 0 or len(bMatrix) == 0:
        print("At least one of your inputted matrices is empty. Please check your input and try again.")
        return -1
    
    if n != len(aMatrix[0]) or n != len(bMatrix):
        print("Size of the A matrix in Ax = b must be n x n (square). Size of the b matrix must be n x 1. Please check your input and try again.")
        return -1
     
    # Check if there exists an entire column of zeroes in aMatrix and return -1 if so.
    columns = map(list, zip(*aMatrix))
    for column in columns:
        if (len(set(column)) != 1) and column[0] == 0:
            print("This problem is unsolvable. There exists at least one column in the A matrix that consists entirely of zeroes.")
            return -1
  
    # Attach bMatrix to the right side of aMatrix so that joinedMatrix = (aMatrix | bMatrix).
    bRow = 0
    joinedMatrix = aMatrix
    for e in joinedMatrix:
        e.append(bMatrix[bRow])
        bRow += 1
    
    # ===== The bulk of the work is done below. =====
    # k should be thought of mostly as the iteration in the algorithm.
    for k in range(n):
        # Find the maximums of the absolute values in each row and bring them to the main diagonal.
        for i in range(k, n):
            if abs(joinedMatrix[i][k]) > abs(joinedMatrix[k][k]):
                joinedMatrix[k], joinedMatrix[i] = joinedMatrix[i], joinedMatrix[k]
            else:
                pass
        
        # Multiply rows by scalars to make the Gaussian Elimination process possible.
        for j in range(k + 1, n):
            s = float(joinedMatrix[j][k] / joinedMatrix[k][k])
            for a in range(k, n+1):
                joinedMatrix[j][a] -= s * joinedMatrix[k][a]
    
    # sol represents the solution matrix.            
    sol = [0 for i in range(n)]
    
    # Solve each row for one particular variable and add that variable to the solution matrix sol.
    sol[n-1] = float(joinedMatrix[n-1][n]) / joinedMatrix[n-1][n-1]
    for i in range(n-1, -1, -1):
        # zeroes used to pad beginning of rows to put solution into row-reduced form.
        zeroes = 0
        for j in range(i + 1, n):
            zeroes = zeroes + float(joinedMatrix[i][j]) * sol[j]
        sol[i] = float(joinedMatrix[i][n] - zeroes) / joinedMatrix[i][i]
    
    return sol