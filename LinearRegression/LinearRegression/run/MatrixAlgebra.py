import sys

class MatrixAlgebra:
    def matrixTranspose(arr1, arr2):
        # iterate arr1 to the length of an item
        for i in range(len(arr1[0])):
            row = []
            for item in arr1:
                # appending to the new arr with values and index position
                # i contains index position and item contains values
                row.append(item[i])
            arr2.append(row)
        return arr2

    def zeroMatrix(rowsLength, colsLength):
        M = []
        while len(M) < rowsLength:
            M.append([])
            while len(M[-1]) < colsLength:
                M[-1].append(0.0)
        return M

    def matrixMultiplication(arr1, arr2):
        rowsArr1 = len(arr1)
        colsArr1 = len(arr1[0])
        rowsArr2 = len(arr2)
        colsArr2 = len(arr2[0])
        if colsArr1 != rowsArr2:
            print("Number of Matrix1 columns does not equal to the number of Matrix2 rows")
            return None

        result = MatrixAlgebra.zeroMatrix(rowsArr1, colsArr2)
        for i in range(rowsArr1):
            for j in range(colsArr2):
                total = 0
                for k in range(colsArr1):
                    total += arr1[i][k] * arr2[k][j]
                result[i][j] = total
        return result

    def getMatrixMinor(arr, i, j):
        return [row[:j] + row[j + 1:] for row in (arr[:i] + arr[i + 1:])]

    def getMatrixDeterminant(arr):
        if len(arr) == 2:
            return arr[0][0] * arr[1][1] - arr[0][1] * arr[1][0]
        determinant = 0
        for col in range(len(arr)):
            determinant += (
                (-1)**col) * arr[0][col] * MatrixAlgebra.getMatrixDeterminant(MatrixAlgebra.getMatrixMinor(arr, 0, col))
        return determinant

    def matrixInverse(arr):
        determinant = MatrixAlgebra.getMatrixDeterminant(arr)

        if len(arr) == 2:
            return [[arr[1][1] / determinant, -1 * arr[0][1] / determinant],
                    [-1 * arr[1][0] / determinant, arr[0][0] / determinant]]

        cofactors = []
        for row in range(len(arr)):
            cofactorRow = []
            for col in range(len(arr)):
                minor = MatrixAlgebra.getMatrixMinor(arr, row, col)
                cofactorRow.append(((-1)**(row + col)) * MatrixAlgebra.getMatrixDeterminant(minor))
            cofactors.append(cofactorRow)
        temp = []
        cofactors = MatrixAlgebra.matrixTranspose(cofactors, temp)
        for row in range(len(cofactors)):
            for col in range(len(cofactors)):
                cofactors[row][col] = cofactors[row][col] / determinant
        return cofactors
