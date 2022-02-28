import math
import cmath


def conjugate(num: (int, float, complex)):
    """
    int -> int
    float -> float
    complex -> complex

    Return the conjugate of a number if it is complex.
    """
    #if input is int or float, return it (no complex conjugate needed)
    if isinstance(num, (int, float)):
        return num
    #if it is complex, subtract twice its imaginary part (effectively switching its sign)
    else:
        result = num - 2j*(num.imag)
        return result

#Matrix error exception, to be used for matrix errors specifically
class MatrixError(Exception):
    pass

class Matrix:
    """
    Matrix object that contains rows and columns of data.
    The data can be accessed by indexing the data attribute:
    Matrix.data[row][column]
    
    Data can be int, float, or complex
    Dimensions of matrix always need to be specified in object creation
    Data is not necessary (it will default to a 0-Matrix)
    Can create an identity matrix with dimensions given if no data is given

    Parameters for object construction: (dimensions:tuple, data:list=[], identity:bool=False)
    Example object creation:
    A = Matrix((2, 3), [[1, 5+1j, 0], [1.12, -3, 43]])     Will create a 2x3 Matrix with that data
    I = Matrix((4,4), identity=True)                       Will create a 4x4 identity matrix
    O = Matrix((2,2))                                      Will create a 2x2 0-matrix [[0,0],[0,0]]
    """
    #take dimensions as tuple, data as list (of lists), identity as bool (shortcut to get identity matrix)
    def __init__(self, dimensions:tuple, data:list=[], identity:bool = False):
        self.dims = dimensions
        self.data = []
        #if data is empty, add 0s in all entries
        if not data:
            for i in range(self.dims[0]):
                self.data.append([0 for j in range(self.dims[1])])
        #if there is data and request identity matrix, raise error
        if data and identity:
            raise MatrixError("Can't pass data and have an indentity matrix.")
        #if want identity, and it's square matrix, make diagonal full of 1s
        #(everything else is already 0)
        if identity and self.dims[0] == self.dims[1]:
            for i in range(self.dims[0]):
                self.data[i][i] = 1
        #identity but not square, raise error
        elif identity:
            raise MatrixError("Identity matrices need to be sqare.")
        #not identity, fill in data from what is given
        elif not identity:
            for row in data:
                #if all are int, float, complex, AND dimensions check out, continue
                if all(isinstance(element, (int, float, complex)) for element in row) and (len(data),len(row)) == self.dims:
                    self.data = data
                #if all are int, float, complex, but dimensions don't check out, raise error
                elif all(isinstance(element, (int, float, complex)) for element in row):
                    raise MatrixError("The data provided does not fit the dimensions given.")
                #if values aren't numbers, raise error
                else:
                    raise MatrixError("Not all data values are numbers.")
        else:
            raise MatrixError("Something went wrong while creating this matrix object.")

    def copy(self):
        """
        Matrix -> Matrix

        Will return a matrix with the same data as the input matrix.
        """
        #empty matrix
        copyMatrix = Matrix(self.dims)
        #copy data
        copyMatrix.data = self.data
        return copyMatrix
                
    def __add__(self, other):
        """
        Matrix + Matrix -> Matrix

        Will return a matrix with each data entry being equal to the sum of the corresponding data
        from the summed matrices.
        """
        if self.dims == other.dims:
            result = Matrix(self.dims)
            for i in range(result.dims[0]):
                #add each corresponding entry
                result.data[i] = [self.data[i][j] + other.data[i][j] for j in range(self.dims[1])]
            return result
        else:
            raise MatrixError("Can't add matrices with different dimensions.")
    
    def __sub__(self, other):
        """
        Matrix - Matrix -> Matrix

        Will return a matrix with each data entry being equal to the difference of the corresponding data
        from the summed matrices.
        """
        if self.dims == other.dims:
            result = Matrix(self.dims)
            for i in range(result.dims[0]):
                #subtract each corresponding entry
                result.data[i] = [self.data[i][j] - other.data[i][j] for j in range(self.dims[1])]
            return result
        else:
            raise MatrixError("Can't subtract matrices with different dimensions.")
    
    def __mul__(self, other):
        """
        Matrix * int,float,complex -> Matrix
        Matrix * Vector -> Vector
        Matrix * Matrix -> Matrix

        Scalars in the form of int, float, or complex will be multiplied to each
        entry in the original matrix, and the new matrix will be returned.

        MxN Matrix and Nx1 Vector multiplication will return an Mx1 Vector object

        MxN Matrix and NxP Matrix multiplication will return an MxP Matrix object
        """
        #scalar-matrix multiplication
        if isinstance(other, (int, float, complex)):
            result = Matrix(self.dims)
            for i in range(self.dims[0]):
                for j in range(self.dims[1]):
                    #multiply each entry by scalar
                    result.data[i][j] = self.data[i][j] * other
            return result
        #vector-matrix multiplication
        elif isinstance(other, Vector) and self.dims[1] == other.dim:
            result = Vector(self.dims[0])
            for i in range(self.dims[0]):
                result.data[i] = sum([self.data[i][j]*other.data[j] for j in range(other.dim)])
            return result
        #matrix-matrix multiplication
        elif self.dims[1] == other.dims[0]:
            result = Matrix((self.dims[0], other.dims[1]))
            for i in range(self.dims[0]):
                for j in range(other.dims[1]):
                    result.data[i][j] = sum([self.data[i][k]*other.data[k][j] for k in range(self.dims[1])])
            return result
        else:
            raise MatrixError("Can't multiply matrices with these dimensions.")
    
    def __truediv__(self, other):
        """
        Matrix / int,float,complex -> Matrix

        Scalars in the form of int, float, or complex will be used as divisors
        to each entry in the original matrix, and the new matrix will be returned.
        """
        if isinstance(other, (int, float, complex)):
            result = Matrix(self.dims)
            for i in range(self.dims[0]):
                for j in range(self.dims[1]):
                    result.data[i][j] = self.data[i][j] / other
            return result
        else:
            raise MatrixError("Can't divide matrices.")
    
    def __pow__(self, power: int):
        """
        Matrix ** int -> Matrix

        Will return the result of multiplying a matrix a certain amount of times
        by itself.
        """
        result = self.copy()
        if power > 0:
            for i in range(power-1):
                result = result * result
            return result
        elif power == 0 and self.dims[0] == self.dims[1]:
            return IdentityMatrix(self.dims[0]).matrix
        else:
            raise MatrixError("Can't raise matrix to that power")

    def __str__(self):
        """
        Matrix -> str

        Will return the Matrix's data in string form (as a list of lists)
        """
        return str(self.data)

    def __iter__(self):
        """
        Matrix -> list

        Will return the Matrix's data as a list of lists
        """
        return iter(self.data)

    def __eq__(self, other):
        """
        Matrix == Matrix -> bool

        Matrices are equal if their data is equal
        """
        return self.data == other.data

    def __ne__(self, other):
        """
        Matrix != Matrix -> bool

        Matrices are not equal if their data is not equal
        """
        return self.data != other.data

    def __abs__(self):
        """
        abs(Matrix) -> int, float, complex
        'Absolute Value' of a Matrix will be its determinant
        """
        return self.det()

    def echelon(self):
        """
        Matrix -> Matrix

        Will perform elementary row operations to a Matrix until it is in
        echelon form (not necessarily reduced echelon form)

        General Echelon Form:
        [*   *   *   *   *]
        [0   *   *   *   *]
        [0   0   *   *   *]
        [0   0   0   0   *]
        where * can be any number.
        """
        global flips
        flips = 0
        global positions
        positions = []

        i = 0
        j = 0
        m = self.dims[0]
        n = self.dims[1]
        while i < m and j < n:
            if self.data[i][j] == 0:
                for row in range(i+1, m):
                    if self.data[row][j] != 0:
                        self.data[i], self.data[row] = self.data[row], self.data[i]
                        flips += 1
                        break
                else:
                    j += 1
                    continue
            for row in range(i+1, m):
                constant = -1 * self.data[row][j] / self.data[i][j]
                multipliedRow = [constant*number for number in self.data[i]]
                self.data[row] = [a + b for a, b in zip(self.data[row], multipliedRow)]
            positions.append([i, j])
            i += 1
            j += 1
        return self

    def det(self):
        """
        Matrix -> int,float,complex

        Will return the determinant of a Matrix object.
        """
        if self.dims[0] != self.dims[1]:
            raise MatrixError("Can't get the determinant of a non-square matrix.")
        global flips
        echMatrix = self.copy().echelon()
        determinant = 1
        for i in range(self.dims[0]):
            determinant *= echMatrix.data[i][i]
        determinant *= (-1)**flips
        return determinant

    def pivotPos(self):
        """
        Matrix -> list

        Will return a list of pivot positions in a Matrix. Each pivot
        position is a list in the form [i,j], indicating there is a
        pivot in the ith row, jth column.
        """
        echMatrix = self.copy().echelon()
        global positions
        return positions
    
    def reduced(self):
        """
        Matrix -> Matrix

        Will return the reduced echelon form of a Matrix by using elementary
        row operations

        General Reduced Echelon Form:
        [1   0   0   *   0]
        [0   1   0   *   0]
        [0   0   1   *   0]
        [0   0   0   0   1]
        where * can be any number
        """
        echMatrix = self.echelon()
        pivotPositions = self.pivotPos()
        for i,j in pivotPositions:
            divConstant = echMatrix.data[i][j]
            echMatrix.data[i] = [value / divConstant for value in echMatrix.data[i]]
            for row in range(0, i):
                multConstant = -1 * echMatrix.data[row][j]
                multipliedRow = [multConstant*number for number in echMatrix.data[i]]
                echMatrix.data[row] = [a + b for a, b in zip(echMatrix.data[row], multipliedRow)]
        return echMatrix

    def transpose(self):
        """
        Matrix -> Matrix

        Will return the Transpose of a Matrix. The Rows and Columns will
        be swapped.

        Example: A 2x3 Matrix -> 3x2 Matrix
        [1   2   3]       [1   4]
        [4   5   6]   ->  [2   5]
                          [3   6]
        """
        matrixT = Matrix((self.dims[1], self.dims[0]))
        for col in range(self.dims[1]):
            newrow = []
            for row in range(self.dims[0]):
                newrow.append(self.data[row][col])
            matrixT.data[col] = newrow
        return matrixT

    def inverse(self):
        """
        Matrix -> Matrix

        Will return the inverse of a Matrix.
        """
        if self.dims[0] != self.dims[1]:
            raise MatrixError("Can't get inverse of non-square matrix.")
        elif self.det() == 0:
            raise MatrixError("Can't get inverse of matrix with determinant 0.")
        matrixCopy = self.copy()
        matrixInv = Matrix(self.dims, identity=True)

        positions = []
        i = 0
        j = 0
        m = matrixCopy.dims[0]
        n = matrixCopy.dims[1]
        while i < m and j < n:
            if matrixCopy.data[i][j] == 0:
                for row in range(i+1, m):
                    if matrixCopy.data[row][j] != 0:
                        matrixCopy.data[i], matrixCopy.data[row] = matrixCopy.data[row], matrixCopy.data[i]
                        matrixInv.data[i], matrixInv.data[row] = matrixInv.data[row], matrixInv.data[i]
                        break
                else:
                    j += 1
                    continue
            for row in range(i+1, m):
                constant = -1 * matrixCopy.data[row][j] / matrixCopy.data[i][j]
                multipliedRow = [constant*number for number in matrixCopy.data[i]]
                matrixCopy.data[row] = [a + b for a, b in zip(matrixCopy.data[row], multipliedRow)]
                multipliedRow = [constant*number for number in matrixInv.data[i]]
                matrixInv.data[row] = [a + b for a, b in zip(matrixInv.data[row], multipliedRow)]
            positions.append([i, j])
            i += 1
            j += 1

        for i,j in positions:
            divConstant = matrixCopy.data[i][j]
            matrixCopy.data[i] = [value / divConstant for value in matrixCopy.data[i]]
            matrixInv.data[i] = [value / divConstant for value in matrixInv.data[i]]
            for row in range(0, i):
                multConstant = -1 * matrixCopy.data[row][j]
                multipliedRow = [multConstant*number for number in matrixCopy.data[i]]
                matrixCopy.data[row] = [a + b for a, b in zip(matrixCopy.data[row], multipliedRow)]
                multipliedRow = [multConstant*number for number in matrixInv.data[i]]
                matrixInv.data[row] = [a + b for a, b in zip(matrixInv.data[row], multipliedRow)]
        return matrixInv
    
    def QR(self):
        """
        Matrix -> Tuple

        Will find the QR Decomposition of a Matrix and return a tuple with
        2 Matrix, Q and R.

        If the initial Matrix is A, then A = QR,
        where Q is an orthogonal matrix, and R is an upper triangular matrix.
        """
        a_i = [Vector(self.dims[1], col) for col in self.transpose()]
        u_i = []
        for i in range(len(a_i)):
            u_i.append(a_i[i])
            for j in range(0, i):
                u_i[i] = u_i[i] - a_i[i].project(u_i[j])
        e_i = [element.unit() for element in u_i]
        Q = Matrix((e_i[0].dim, len(e_i)), [element.data for element in e_i]).transpose()
        R = Matrix((e_i[0].dim, len(e_i)))
        for i in range(e_i[0].dim):
            for j in range(i, len(e_i)):
                innerProduct = sum([a_i[j].data[k] * conjugate(e_i[i].data[k]) for k in range(e_i[i].dim)])
                R.data[i][j] = innerProduct
        return (Q, R)

    def eigenvalues(self):
        """
        Matrix -> list

        Will return a list of the eigenvalues of the matrix.
        """
        A_k = self.copy()
        for m in range(A_k.dims[0]-1, 0, -1):
            k = 0
            while k < 1000:
                sigma_k = A_k.data[m][m]
                Q, R = (A_k - (Matrix((A_k.dims), identity=True)*sigma_k)).QR()
                A_kPrev = A_k
                A_k = (R*Q) + (Matrix((A_k.dims), identity=True)*sigma_k)
                stable = True
                for i in range(A_k.dims[0]):
                    if A_k.data[i][i] == 0 or A_kPrev.data[i][i] == 0:
                        continue
                    if abs(A_k.data[i][i] / A_kPrev.data[i][i]) < 0.999 or abs(A_k.data[i][i] / A_kPrev.data[i][i]) > 1.001:
                        stable = False
                        break
                if stable:
                    break
                k += 1
    
        eValues = []
        i = 0
        while i < A_k.dims[0]-1:
            if abs(A_k.data[i+1][i] / A_k.data[i][i]) > 0.001:
                a = A_k.data[i][i]
                b = A_k.data[i][i+1]
                c = A_k.data[i+1][i]
                d = A_k.data[i+1][i+1]
                complexValA = (a + d + cmath.sqrt(a**2 + 2*a*d + d**2 + 4*b*c - 4*d*a))/2
                complexValB = (a + d - cmath.sqrt(a**2 + 2*a*d + d**2 + 4*b*c - 4*d*a))/2
                eValues.append(complexValA)
                eValues.append(complexValB)
                i += 2
            else:
                eValues.append(A_k.data[i][i])
                i += 1
        if len(eValues) < A_k.dims[0]:
            eValues.append(A_k.data[-1][-1])
        return eValues


















class VectorError(Exception):
    pass

class Vector:
    """
    Vector object that contains data in a list.
    The data can be accessed by indexing the data attribute:
    Vector.data[index]
    
    Data can be int, float, or complex
    Dimension / Length of Vector always need to be specified in object creation
    Data is not necessary (it will default to a 0-Vector)

    Parameters for object construction: (dimension:int, data:list=[])
    Example object creation:
    v = Vector(3, [1, 5+1j, 0])     Will create a 3-entry vector with that data
    w = Vector(2)                   Will create a 2-entry vector with all 0s [0, 0]
    """
    def __init__(self, dimension:int, data:list=[]):
        self.dim = dimension
        self.data = []
        if not data:
            self.data = [0 for i in range(self.dim)]
        elif data:
            if all(isinstance(element, (int, float, complex)) for element in data) and len(data) == self.dim:
                self.data = data
            elif all(isinstance(element, (int, float, complex)) for element in data):
                raise VectorError("The data provided does not fit the dimension given.")
            else:
                raise VectorError("Not all data values are numbers.")
        else:
            raise VectorError("Something went wrong while creating this vector object.")

    def copy(self):
        """
        Vector -> Vector

        Will return a vector with the same data as the input vector.
        """
        copyVector = Vector(self.dim)
        copyVector.data = self.data
        return copyVector
                
    def __add__(self, other):
        """
        Vector + Vector -> Vector

        Will return a vector with each data entry being equal to the sum of the corresponding data
        from the summed vectors.
        """
        if self.dim == other.dim:
            result = Vector(self.dim)
            result.data = [self.data[i] + other.data[i] for i in range(self.dim)]
            return result
        else:
            raise VectorError("Can't add vectors with different dimensions.")
    
    def __sub__(self, other):
        """
        Vector - Vector -> Vector

        Will return a vector with each data entry being equal to the difference of the corresponding data
        from the summed vectors.
        """
        if self.dim == other.dim:
            result = Vector(self.dim)
            result.data = [self.data[i] - other.data[i] for i in range(self.dim)]
            return result
        else:
            raise VectorError("Can't add vectors with different dimensions.")
    
    def __mul__(self, other):
        """
        Vector * int,float,complex -> Vector
        Vector * Vector -> int, float, complex

        Scalars in the form of int, float, or complex will be multiplied to each
        entry in the original vector, and the new vector will be returned.

        Two vectors of the same dimension can multiply and return their dot product.
        """
        if isinstance(other, (int, float, complex)):
            result = Vector(self.dim)
            result.data = [self.data[i] * other for i in range(self.dim)]
            return result
        elif self.dim == other.dim:
            result = sum([self.data[i] * other.data[i] for i in range(self.dim)])
            return result
        else:
            raise VectorError("Can't dot product vectors with these dimensions.")

    def __truediv__(self, other):
        """
        Vector / int,float,complex -> Vector

        Scalars in the form of int, float, or complex will be used as divisors
        to each entry in the original vector, and the new vector will be returned.
        """
        if isinstance(other, (int, float, complex)):
            result = Vector(self.dim)
            result.data = [self.data[i] / other for i in range(self.dim)]
            return result
        else:
            raise VectorError("Can't divide vectors.")

    def __str__(self):
        """
        Vector -> str

        Will return the Vector's data in string form (as a list of lists)
        """
        return str(self.data)

    def __iter__(self):
        """
        Vector -> list

        Will return the vector's data as a list of lists
        """
        return iter(self.data)

    def __eq__(self, other):
        """
        Vector == Vector -> bool

        Vectors are equal if their data is equal
        """
        return self.data == other.data

    def __ne__(self, other):
        """
        Vector == Vector -> bool

        Vectors are not equal if their data is not equal
        """
        return self.data != other.data

    def __abs__(self):
        """
        abs(Vector) -> int, float

        Will return the magnitude of the Vector.
        """
        return self.magnitude()

    def cross(self, other):
        """
        Vector.cross(Vector) -> Vector

        Will return the vector cross product of 2 3-dimensional vectors.
        """
        if self.dim != 3 or other.dim != 3:
            raise VectorError("Can't cross product vectors that are not 3-dimensional.")
        xValue = self.data[2]*other.data[3] - self.data[3]*other.data[2]
        yValue = self.data[3]*other.data[1] - self.data[1]*other.data[3]
        zValue = self.data[1]*other.data[2] - self.data[2]*other.data[1]
        return Vector(3, [xValue, yValue, zValue])

    def magnitude(self):
        """
        Vector -> int, float

        Will return the magnitude of a vector.
        """
        return round(math.sqrt(sum([(abs(num))**2 for num in self.data])), 4)

    def project(self, other):
        """
        Vector1.project(Vector2) -> Vector

        Will return a vector that is often defined as the projection of Vector1 onto Vector2
        """
        if self.dim != other.dim:
            raise VectorError("Can't project vector into another vector of different dimension.")
        if other.magnitude() == 0:
            return Vector(self.dim, data=[])
        if all(isinstance(num, (int, float)) for num in self.data + other.data):
            scalar = (self*other) / (other*other)
            data = [scalar*num for num in other.data]
            return Vector(self.dim, data)
        else:
            numerator = sum([self.data[i] * conjugate(other.data[i]) for i in range(self.dim)])
            denominator = sum([conjugate(other.data[i]) * other.data[i] for i in range(self.dim)])
            scalar = numerator / denominator
            data = [scalar*num for num in other.data]
            return Vector(self.dim, data)
    
    def unit(self):
        """
        Vector -> Vector

        Will return the unit-vector version of the original vector, by scaling it
        to have a magnitude of 1.
        """
        result = self.copy()
        if result.magnitude() == 0:
            return result
        return result / result.magnitude()
