//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_exec_math_Matrix_h
#define __dax_exec_math_Matrix_h

#include <dax/Types.h>
#include <dax/TypeTraits.h>
#include <dax/VectorTraits.h>

#include <dax/exec/math/Precision.h>
#include <dax/exec/math/Sign.h>

namespace dax {
namespace exec {
namespace math {

// Making non-square matricies may be overkill.

// If matricies are really useful, they may be promoted to dax/Types.h (and
// the dax namespace).

/// Basic Matrix type.
///
template<typename T, int NumRow, int NumCol>
class Matrix {
public:
  typedef T ComponentType;
  static const int NUM_ROWS = NumRow;
  static const int NUM_COLUMNS = NumCol;

  DAX_EXEC_EXPORT Matrix() { }
  DAX_EXEC_EXPORT explicit Matrix(const ComponentType &value)
    : Components(dax::Tuple<ComponentType, NUM_COLUMNS>(value)) { }

  /// Brackets are used to reference a matrix like a 2D array (i.e.
  /// matrix[row][column]).
  DAX_EXEC_EXPORT
  const dax::Tuple<ComponentType, NUM_COLUMNS> &operator[](int rowIndex) const {
    return this->Components[rowIndex];
  }
  /// Brackets are used to referens a matrix like a 2D array i.e.
  /// matrix[row][column].
  DAX_EXEC_EXPORT
  dax::Tuple<ComponentType, NUM_COLUMNS> &operator[](int rowIndex) {
    return this->Components[rowIndex];
  }

  /// Parentheses are used to reference a matrix using mathematical tuple
  /// notation i.e. matrix(row,column).
  DAX_EXEC_EXPORT
  const ComponentType &operator()(int rowIndex, int colIndex) const {
    return (*this)[rowIndex][colIndex];
  }
  /// Parentheses are used to reference a matrix using mathematical tuple
  /// notation i.e. matrix(row,column).
  DAX_EXEC_EXPORT
  ComponentType &operator()(int rowIndex, int colIndex) {
    return (*this)[rowIndex][colIndex];
  }

private:
  dax::Tuple<dax::Tuple<ComponentType, NUM_COLUMNS>, NUM_ROWS> Components;
};

/// Returns a tuple containing the given row (indexed from 0) of the given
/// matrix.
///
template<typename T, int NumRow, int NumCol>
DAX_EXEC_EXPORT const dax::Tuple<T, NumCol> &MatrixRow(
    const dax::exec::math::Matrix<T,NumRow,NumCol> &matrix, int rowIndex)
{
  return matrix[rowIndex];
}

/// Returns a tuple containing the given column (indexed from 0) of the given
/// matrix.  Might not be as efficient as the Row function.
///
template<typename T, int NumRow, int NumCol>
DAX_EXEC_EXPORT dax::Tuple<T, NumRow> MatrixColumn(
    const dax::exec::math::Matrix<T,NumRow,NumCol> &matrix, int columnIndex)
{
  dax::Tuple<T, NumRow> columnValues;
  for (int rowIndex = 0; rowIndex < NumRow; rowIndex++)
    {
    columnValues[rowIndex] = matrix(rowIndex, columnIndex);
    }
  return columnValues;
}

/// Convenience function for setting a row of a matrix.
///
template<typename T, int NumRow, int NumCol>
DAX_EXEC_EXPORT
void MatrixSetRow(dax::exec::math::Matrix<T,NumRow,NumCol> &matrix,
                  int rowIndex,
                  dax::Tuple<T,NumCol> rowValues)
{
  matrix[rowIndex] = rowValues;
}

/// Convenience function for setting a column of a matrix.
///
template<typename T, int NumRow, int NumCol>
DAX_EXEC_EXPORT
void MatrixSetColumn(dax::exec::math::Matrix<T,NumRow,NumCol> &matrix,
                     int columnIndex,
                     dax::Tuple<T,NumRow> columnValues)
{
  for (int rowIndex = 0; rowIndex < NumRow; rowIndex++)
    {
    matrix(rowIndex, columnIndex) = columnValues[rowIndex];
    }
}

/// Standard matrix multiplication.
///
template<typename T, int NumRow, int NumCol, int NumInternal>
DAX_EXEC_EXPORT
dax::exec::math::Matrix<T,NumRow,NumCol> MatrixMultiply(
    const dax::exec::math::Matrix<T,NumRow,NumInternal> &leftFactor,
    const dax::exec::math::Matrix<T,NumInternal,NumCol> &rightFactor)
{
  dax::exec::math::Matrix<T,NumRow,NumCol> result;
  for (int rowIndex = 0; rowIndex < NumRow; rowIndex++)
    {
    for (int colIndex = 0; colIndex < NumCol; colIndex++)
      {
      T sum = leftFactor(rowIndex, 0) * rightFactor(0, colIndex);
      for (int internalIndex = 1; internalIndex < NumInternal; internalIndex++)
        {
        sum += leftFactor(rowIndex, internalIndex)
            * rightFactor(internalIndex, colIndex);
        }
      result(rowIndex, colIndex) = sum;
      }
    }
  return result;
}

/// Standard matrix-vector multiplication.
///
template<typename T, int NumRow, int NumCol>
DAX_EXEC_EXPORT
dax::Tuple<T,NumRow> MatrixMultiply(
    const dax::exec::math::Matrix<T,NumRow,NumCol> &leftFactor,
    const dax::Tuple<T,NumCol> &rightFactor)
{
  dax::Tuple<T,NumRow> product;
  for (int rowIndex = 0; rowIndex < NumRow; rowIndex++)
    {
    product[rowIndex] =
        dax::dot(dax::exec::math::MatrixRow(leftFactor,rowIndex), rightFactor);
    }
  return product;
}

/// Standard vector-matrix multiplication
///
template<typename T, int NumRow, int NumCol>
DAX_EXEC_EXPORT
dax::Tuple<T,NumCol> MatrixMultiply(
    const dax::Tuple<T,NumRow> &leftFactor,
    const dax::exec::math::Matrix<T,NumRow,NumCol> &rightFactor)
{
  dax::Tuple<T,NumCol> product;
  for (int colIndex = 0; colIndex < NumCol; colIndex++)
    {
    product[colIndex] =
        dax::dot(leftFactor,
                 dax::exec::math::MatrixColumn(rightFactor, colIndex));
    }
  return product;
}

/// Returns the identity matrix.
///
template<typename T, int Size>
DAX_EXEC_EXPORT
dax::exec::math::Matrix<T,Size,Size> MatrixIdentity()
{
  dax::exec::math::Matrix<T,Size,Size> result(0);
  for (int index = 0; index < Size; index++)
    {
    result(index,index) = 1.0;
    }
  return result;
}

/// Fills the given matrix with the identity matrix.
///
template<typename T, int Size>
DAX_EXEC_EXPORT
void MatrixIdentity(dax::exec::math::Matrix<T,Size,Size> &matrix)
{
  matrix = dax::exec::math::MatrixIdentity<T,Size>();
}

/// Returns the transpose of the given matrix.
///
template<typename T, int NumRows, int NumCols>
DAX_EXEC_EXPORT
dax::exec::math::Matrix<T,NumCols,NumRows> MatrixTranspose(
    const dax::exec::math::Matrix<T,NumRows,NumCols> &matrix)
{
  dax::exec::math::Matrix<T,NumCols,NumRows> result;
  for (int index = 0; index < NumRows; index++)
    {
    dax::exec::math::MatrixSetColumn(result, index, matrix[index]);
    }
  return result;
}

namespace detail {

// Used with MatrixLUPFactor.
template<int Size>
DAX_EXEC_EXPORT
void MatrixLUPFactorFindPivot(dax::exec::math::Matrix<dax::Scalar,Size,Size> &A,
                              dax::Tuple<int,Size> &permutation,
                              int topCornerIndex,
                              dax::Scalar &inversionParity,
                              bool &valid)
{
  int maxRowIndex = topCornerIndex;
  dax::Scalar maxValue = dax::exec::math::Abs(A(maxRowIndex, topCornerIndex));
  for (int rowIndex = topCornerIndex + 1; rowIndex < Size; rowIndex++)
    {
    dax::Scalar compareValue =
        dax::exec::math::Abs(A(rowIndex, topCornerIndex));
    if (maxValue < compareValue)
      {
      maxValue = compareValue;
      maxRowIndex = rowIndex;
      }
    }

  if (maxValue < dax::exec::math::Epsilon()) { valid = false; }

  if (maxRowIndex != topCornerIndex)
    {
    // Swap rows in matrix.
    dax::Tuple<dax::Scalar,Size> maxRow =
        dax::exec::math::MatrixRow(A, maxRowIndex);
    dax::exec::math::MatrixSetRow(A,
                                  maxRowIndex,
                                  dax::exec::math::MatrixRow(A,topCornerIndex));
    dax::exec::math::MatrixSetRow(A, topCornerIndex, maxRow);

    // Record change in permutation matrix.
    int maxOriginalRowIndex = permutation[maxRowIndex];
    permutation[maxRowIndex] = permutation[topCornerIndex];
    permutation[topCornerIndex] = maxOriginalRowIndex;

    // Keep track of inversion parity.
    inversionParity = -inversionParity;
    }
}

// Used with MatrixLUPFactor
template<int Size>
DAX_EXEC_EXPORT
void MatrixLUPFactorFindUpperTriangleElements(
    dax::exec::math::Matrix<dax::Scalar,Size,Size> &A,
    int topCornerIndex)
{
  // Compute values for upper triangle on row topCornerIndex
  for (int colIndex = topCornerIndex+1; colIndex < Size; colIndex++)
    {
    A(topCornerIndex,colIndex) /= A(topCornerIndex,topCornerIndex);
    }

  // Update the rest of the matrix for calculations on subsequent rows
  for (int rowIndex = topCornerIndex+1; rowIndex < Size; rowIndex++)
    {
    for (int colIndex = topCornerIndex+1; colIndex < Size; colIndex++)
      {
      A(rowIndex,colIndex) -=
          A(rowIndex,topCornerIndex)*A(topCornerIndex,colIndex);
      }
    }
}

/// Performs an LUP-factorization on the given matrix using Crout's method. The
/// LU-factorization takes a matrix A and decomposes it into a lower triangular
/// matrix L and upper triangular matrix U such that A = LU. The
/// LUP-factorization also allows permutation of A, which makes the
/// decomposition always posible so long as A is not singular. In addition to
/// matrices L and U, LUP also finds permutation matrix P containing all zeros
/// except one 1 per row and column such that PA = LU.
///
/// The result is done in place such that the lower triangular matrix, L, is
/// stored in the lower-left triangle of A including the diagonal. The upper
/// triangular matrix, U, is stored in the upper-right triangle of L not
/// including the diagonal. The diagonal of U in Crout's method is all 1's (and
/// therefore not explicitly stored).
///
/// The permutation matrix P is represented by the permutation vector. If
/// permutation[i] = j then row j in the original matrix A has been moved to
/// row i in the resulting matrices. The permutation matrix P can be
/// represented by a matrix with p_i,j = 1 if permutation[i] = j and 0
/// otherwise. If using LUP-factorization to compute a determinant, you also
/// need to know the parity (whether there is an odd or even amount) of
/// inversions. An inversion is an instance of a smaller number appearing after
/// a larger number in permutation. Although you can compute the inversion
/// parity after the fact, this function keeps track of it with much less
/// compute resources. The parameter inversionParity is set to 1.0 for even
/// parity and -1.0 for odd parity.
///
/// Not all matrices (specifically singular matrices) have an
/// LUP-factorization. If the LUP-factorization succeeds, valid is set to true.
/// Otherwise, valid is set to false and the result is indeterminant.
///
template<int Size>
DAX_EXEC_EXPORT
void MatrixLUPFactor(dax::exec::math::Matrix<dax::Scalar,Size,Size> &A,
                     dax::Tuple<int,Size> &permutation,
                     dax::Scalar &inversionParity,
                     bool &valid)
{
  // Initialize permutation.
  for (int index = 0; index < Size; index++) { permutation[index] = index; }
  inversionParity = 1;
  valid = true;

  for (int rowIndex = 0; rowIndex < Size; rowIndex++)
    {
    MatrixLUPFactorFindPivot(A, permutation, rowIndex, inversionParity, valid);
    MatrixLUPFactorFindUpperTriangleElements(A, rowIndex);
    }
}

/// Use a previous factorization done with MatrixLUPFactor to solve the
/// system Ax = b.  Instead of A, this method takes in the LU and P
/// matrices calculated by MatrixLUPFactor from A.
///
template<int Size>
DAX_EXEC_EXPORT
void MatrixLUPSolve(const dax::exec::math::Matrix<dax::Scalar,Size,Size> &LU,
                    const dax::Tuple<int,Size> &permutation,
                    const dax::Tuple<dax::Scalar,Size> &b,
                    dax::Tuple<dax::Scalar,Size> &x)
{
  // The LUP-factorization gives us PA = LU or equivalently A = inv(P)LU.
  // Substituting into Ax = b gives us inv(P)LUx = b or LUx = Pb.
  // Now consider the intermediate vector y = Ux.
  // Substituting in the previous two equations yields Ly = Pb.
  // Solving Ly = Pb is easy because L is triangular and P is just a
  // permutation.
  dax::Tuple<dax::Scalar,Size> y;
  for (int rowIndex = 0; rowIndex < Size; rowIndex++)
    {
    y[rowIndex] = b[permutation[rowIndex]];
    // Recall that L is stored in the lower triangle of LU including diagonal.
    for (int colIndex = 0; colIndex < rowIndex; colIndex++)
      {
      y[rowIndex] -= LU(rowIndex,colIndex)*y[colIndex];
      }
    y[rowIndex] /= LU(rowIndex,rowIndex);
    }

  // Now that we have y, we can easily solve Ux = y for x.
  for (int rowIndex = Size-1; rowIndex >= 0; rowIndex--)
    {
    x[rowIndex] = y[rowIndex];
    // Recall that U is stored in the upper triangle of LU with the diagonal
    // implicitly all 1's.
    for (int colIndex = rowIndex+1; colIndex < Size; colIndex++)
      {
      x[rowIndex] -= LU(rowIndex,colIndex)*x[colIndex];
      }
    }
}

} // namespace detail

/// Solve the linear system Ax = b for x. If a single solution is found, valid
/// is set to true, false otherwise.
///
template<int Size>
DAX_EXEC_EXPORT
dax::Tuple<dax::Scalar,Size> SolveLinearSystem(
    const dax::exec::math::Matrix<dax::Scalar,Size,Size> &A,
    const dax::Tuple<dax::Scalar,Size> &b,
    bool &valid)
{
  // First, we will make an LUP-factorization to help us.
  dax::exec::math::Matrix<dax::Scalar,Size,Size> LU = A;
  dax::Tuple<int,Size> permutation;
  dax::Scalar inversionParity;  // Unused.
  dax::exec::math::detail::MatrixLUPFactor(LU,
                                           permutation,
                                           inversionParity,
                                           valid);

  // Next, use the decomposition to solve the system.
  dax::Tuple<dax::Scalar,Size> x;
  dax::exec::math::detail::MatrixLUPSolve(LU, permutation, b, x);
  return x;
}

/// Find and return the inverse of the given matrix. If the matrix is singular,
/// the inverse will not be correct and valid will be set to false.
///
template<int Size>
DAX_EXEC_EXPORT
dax::exec::math::Matrix<dax::Scalar,Size,Size> MatrixInverse(
    const dax::exec::math::Matrix<dax::Scalar,Size,Size> &A,
    bool &valid)
{
  // First, we will make an LUP-factorization to help us.
  dax::exec::math::Matrix<dax::Scalar,Size,Size> LU = A;
  dax::Tuple<int,Size> permutation;
  dax::Scalar inversionParity;  // Unused
  dax::exec::math::detail::MatrixLUPFactor(LU,
                                           permutation,
                                           inversionParity,
                                           valid);

  // We will use the decomposition to solve AX = I for X where X is
  // clearly the inverse of A.  Our solve method only works for vectors,
  // so we solve for one column of invA at a time.
  dax::exec::math::Matrix<dax::Scalar,Size,Size> invA;
  dax::Tuple<dax::Scalar,Size> ICol(dax::Scalar(0));
  dax::Tuple<dax::Scalar,Size> invACol;
  for (int colIndex = 0; colIndex < Size; colIndex++)
    {
    ICol[colIndex] = 1;
    dax::exec::math::detail::MatrixLUPSolve(LU, permutation, ICol, invACol);
    ICol[colIndex] = 0;
    dax::exec::math::MatrixSetColumn(invA, colIndex, invACol);
    }
  return invA;
}

/// Compute the determinant of a matrix.
///
template<int Size>
DAX_EXEC_EXPORT
dax::Scalar MatrixDeterminant(
    const dax::exec::math::Matrix<dax::Scalar,Size,Size> &A)
{
  // First, we will make an LUP-factorization to help us.
  dax::exec::math::Matrix<dax::Scalar,Size,Size> LU = A;
  dax::Tuple<int,Size> permutation;
  dax::Scalar inversionParity;
  bool valid;
  dax::exec::math::detail::MatrixLUPFactor(LU,
                                           permutation,
                                           inversionParity,
                                           valid);

  // If the matrix is singular, the factorization is invalid, but in that
  // case we know that the determinant is 0.
  if (!valid) { return 0; }

  // The determinant is equal to the product of the diagonal of the L matrix,
  // possibly negated depending on the parity of the inversion. The
  // inversionParity variable is set to 1.0 and -1.0 for even and odd parity,
  // respectively. This sign determines whether the product should be negated.
  dax::Scalar product = inversionParity;
  for (int index = 0; index < Size; index++)
    {
    product *= LU(index,index);
    }
  return product;
}

// Specializations for common small determinants.

template<>
DAX_EXEC_EXPORT
dax::Scalar MatrixDeterminant<1>(
    const dax::exec::math::Matrix<dax::Scalar,1,1> &A)
{
  return A(0,0);
}

template<>
DAX_EXEC_EXPORT
dax::Scalar MatrixDeterminant<2>(
    const dax::exec::math::Matrix<dax::Scalar,2,2> &A)
{
  return A(0,0)*A(1,1) - A(1,0)*A(0,1);
}

template<>
DAX_EXEC_EXPORT
dax::Scalar MatrixDeterminant<3>(
    const dax::exec::math::Matrix<dax::Scalar,3,3> &A)
{
  return A(0,0) * A(1,1) * A(2,2) + A(1,0) * A(2,1) * A(0,2) +
         A(2,0) * A(0,1) * A(1,2) - A(0,0) * A(2,1) * A(1,2) -
         A(1,0) * A(0,1) * A(2,2) - A(2,0) * A(1,1) * A(0,2);
}

}
}
} // namespace dax::exec::math

// Implementations of traits for matrices.

namespace dax {

/// Tag used to identify 2 dimensional types (matrices). A TypeTraits class
/// will typedef this class to DimensionalityTag.
///
struct TypeTraitsMatrixTag {};

template<typename T, int NumRow, int NumCol>
struct TypeTraits<dax::exec::math::Matrix<T, NumRow, NumCol> > {
  typedef typename TypeTraits<T>::NumericTag NumericTag;
  typedef TypeTraitsMatrixTag DimensionalityTag;
};

/// A matrix has vector traits to implement component-wise operations.
///
template<typename T, int NumRow, int NumCol>
struct VectorTraits<dax::exec::math::Matrix<T, NumRow, NumCol> > {
private:
  typedef dax::exec::math::Matrix<T, NumRow, NumCol> MatrixType;
public:
  typedef T ComponentType;
  static const int NUM_COMPONENTS = NumRow*NumCol;
  typedef dax::VectorTraitsTagMultipleComponents HasMultipleComponents;

  DAX_EXEC_EXPORT static const ComponentType &GetComponent(
      const MatrixType &matrix, int component) {
    int colIndex = component % NumCol;
    int rowIndex = component / NumCol;
    return matrix(rowIndex,colIndex);
  }
  DAX_EXEC_EXPORT static ComponentType &GetComponent(
      MatrixType &matrix, int component) {
    int colIndex = component % NumCol;
    int rowIndex = component / NumCol;
    return matrix(rowIndex,colIndex);
  }
  DAX_EXEC_EXPORT static void SetComponent(MatrixType &matrix,
                                           int component,
                                           T value)
  {
    GetComponent(matrix, component) = value;
  }
};

} // namespace dax

// Basic comparison operators.

template<typename T, int NumRow, int NumCol>
DAX_EXEC_EXPORT bool operator==(
    const dax::exec::math::Matrix<T,NumRow,NumCol> &a,
    const dax::exec::math::Matrix<T,NumRow,NumCol> &b)
{
  for (int colIndex = 0; colIndex < NumCol; colIndex++)
    {
    for (int rowIndex = 0; rowIndex < NumRow; rowIndex++)
      {
      if (a(rowIndex, colIndex) != b(rowIndex, colIndex)) return false;
      }
    }
  return true;
}
template<typename T, int NumRow, int NumCol>
DAX_EXEC_EXPORT bool operator!=(
    const dax::exec::math::Matrix<T,NumRow,NumCol> &a,
    const dax::exec::math::Matrix<T,NumRow,NumCol> &b)
{
  return !(a == b);
}

#endif //__dax_exec_math_Matrix_h
