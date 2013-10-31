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

#include <dax/math/Matrix.h>

#include <dax/VectorTraits.h>

#include <dax/testing/Testing.h>

namespace {

#define FOR_ROW_COL(matrix) \
  for(int row=0; row < (matrix).NUM_ROWS; row++) \
  for(int col=0; col < (matrix).NUM_COLUMNS; col++)

template<typename T, int NumRow, int NumCol>
struct MatrixTest
{
  static const int NUM_ROWS = NumRow;
  static const int NUM_COLS = NumCol;
  typedef dax::math::Matrix<T,NUM_ROWS,NUM_COLS> MatrixType;

  static inline T value(int row, int col)
  {
    return static_cast<T>(row + 100*col);
  }

  static void BasicCreation()
  {
    std::cout << "Basic creation." << std::endl;
    MatrixType matrix(5);
    FOR_ROW_COL(matrix)
      {
      DAX_TEST_ASSERT(matrix(row,col) == static_cast<T>(5),
                      "Constant set incorrect.");
      }
  }

  static void BasicAccessors()
  {
    std::cout << "Basic accessors." << std::endl;
    MatrixType matrix;
    FOR_ROW_COL(matrix)
      {
      matrix[row][col] = value(row,col)*2;
      }
    FOR_ROW_COL(matrix)
      {
      DAX_TEST_ASSERT(matrix(row,col) == value(row,col)*2,
                      "Bad set or retreive.");
      const MatrixType const_matrix = matrix;
      DAX_TEST_ASSERT(const_matrix(row,col) == value(row,col)*2,
                      "Bad set or retreive.");
      }

    FOR_ROW_COL(matrix)
      {
      matrix(row,col) = value(row,col);
      }
    const MatrixType const_matrix = matrix;
    FOR_ROW_COL(matrix)
      {
      DAX_TEST_ASSERT(matrix[row][col] == value(row,col),
                      "Bad set or retreive.");
      DAX_TEST_ASSERT(const_matrix[row][col] == value(row,col),
                      "Bad set or retreive.");
      }
    DAX_TEST_ASSERT(matrix == const_matrix,
                    "Equal test operator not working.");
    DAX_TEST_ASSERT(!(matrix != const_matrix),
                    "Not-Equal test operator not working.");
    DAX_TEST_ASSERT(test_equal(matrix, const_matrix),
                    "Vector-based equal test not working.");
  }

  static MatrixType CreateMatrix()
  {
    MatrixType matrix;
    FOR_ROW_COL(matrix)
      {
      matrix(row,col) = value(row,col);
      }
    return matrix;
  }

  static void RowColAccessors()
  {
    typedef dax::Tuple<T,NUM_ROWS> ColumnType;
    typedef dax::Tuple<T,NUM_COLS> RowType;
    const MatrixType const_matrix = CreateMatrix();
    MatrixType matrix;

    std::cout << "Access by row or column" << std::endl;
    FOR_ROW_COL(matrix)
      {
      RowType rowvec = dax::math::MatrixRow(const_matrix, row);
      DAX_TEST_ASSERT(rowvec[col] == const_matrix(row,col), "Bad get row.");
      ColumnType columnvec = dax::math::MatrixColumn(const_matrix, col);
      DAX_TEST_ASSERT(columnvec[row] == const_matrix(row,col), "Bad get col.");
      }

    std::cout << "Set a row." << std::endl;
    for (int row = 0; row < NUM_ROWS; row++)
      {
      RowType rowvec = dax::math::MatrixRow(const_matrix, NUM_ROWS-row-1);
      dax::math::MatrixSetRow(matrix, row, rowvec);
      }
    FOR_ROW_COL(matrix)
      {
      DAX_TEST_ASSERT(matrix(NUM_ROWS-row-1,col) == const_matrix(row,col),
                      "Rows not set right.");
      }

    std::cout << "Set a column." << std::endl;
    for (int col = 0; col < NUM_COLS; col++)
      {
      ColumnType colvec =
          dax::math::MatrixColumn(const_matrix, NUM_COLS-col-1);
      dax::math::MatrixSetColumn(matrix, col, colvec);
      }
    FOR_ROW_COL(matrix)
      {
      DAX_TEST_ASSERT(matrix(row,NUM_COLS-col-1) == const_matrix(row,col),
                      "Columns not set right.");
      }
  }

  static void Multiply()
  {
    std::cout << "Try multiply." << std::endl;
    const MatrixType leftFactor = CreateMatrix();
    dax::math::Matrix<T,NUM_COLS,4> rightFactor;
    for (int index = 0; index < NUM_COLS*4; index++)
      {
      dax::VectorTraits<dax::math::Matrix<T,NUM_COLS,4> >::SetComponent(
            rightFactor, index, index);
      }

    dax::math::Matrix<T,NUM_ROWS,4> product
        = dax::math::MatrixMultiply(leftFactor, rightFactor);

    FOR_ROW_COL(product)
      {
      dax::Tuple<T,NUM_COLS> leftVector
          = dax::math::MatrixRow(leftFactor, row);
      dax::Tuple<T,NUM_COLS> rightVector
          = dax::math::MatrixColumn(rightFactor, col);
      DAX_TEST_ASSERT(test_equal(product(row,col),
                                 dax::dot(leftVector,rightVector)),
                      "Matrix multiple wrong.");
      }

    std::cout << "Vector multiply." << std::endl;
    MatrixType matrixFactor;
    dax::Tuple<T,NUM_ROWS> leftVector(2);
    dax::Tuple<T,NUM_COLS> rightVector;
    FOR_ROW_COL(matrixFactor)
      {
      matrixFactor(row,col) = row + 1;
      rightVector[col] = col + 1;
      }

    dax::Tuple<T,NUM_COLS> leftResult =
        dax::math::MatrixMultiply(leftVector, matrixFactor);
    for (int index = 0; index < NUM_COLS; index++)
      {
      DAX_TEST_ASSERT(test_equal(leftResult[index], T(NUM_ROWS*(NUM_ROWS+1))),
                      "Vector/matrix multiple wrong.");
      }

    dax::Tuple<T,NUM_ROWS> rightResult =
        dax::math::MatrixMultiply(matrixFactor, rightVector);
    for (int index = 0; index < NUM_ROWS; index++)
      {
      DAX_TEST_ASSERT(test_equal(rightResult[index],
                                 T(((index+1)*NUM_COLS*(NUM_COLS+1))/2)),
                      "Matrix/vector multiple wrong.");
      }
  }

  static void Identity()
  {
    std::cout << "Check identity" << std::endl;

    MatrixType originalMatrix = CreateMatrix();

    dax::math::Matrix<T,NUM_COLS,NUM_COLS> identityMatrix;
    dax::math::MatrixIdentity(identityMatrix);

    MatrixType multMatrix =
        dax::math::MatrixMultiply(originalMatrix, identityMatrix);

    DAX_TEST_ASSERT(test_equal(originalMatrix, multMatrix),
                    "Identity is not really identity.");

  }

  static void Transpose()
  {
    std::cout << "Check transpose" << std::endl;

    MatrixType originalMatrix = CreateMatrix();

    dax::math::Matrix<T,NUM_COLS,NUM_ROWS> transMatrix =
        dax::math::MatrixTranspose(originalMatrix);
    FOR_ROW_COL(originalMatrix)
      {
      DAX_TEST_ASSERT(originalMatrix(row,col) == transMatrix(col,row),
                      "Transpose wrong.");
      }
  }

  static void Run()
  {
  std::cout << "-- " << NUM_ROWS << " x " << NUM_COLS << std::endl;

  BasicCreation();
  BasicAccessors();
  RowColAccessors();
  Multiply();
  Identity();
  Transpose();
  }

private:
  MatrixTest(); // Not implemented
};

template<typename T, int NumRow>
void MatrixTest1()
{
  MatrixTest<T,NumRow,1>::Run();
  MatrixTest<T,NumRow,2>::Run();
  MatrixTest<T,NumRow,3>::Run();
  MatrixTest<T,NumRow,4>::Run();
  MatrixTest<T,NumRow,5>::Run();
}

template<int Size>
void NonSingularMatrix(dax::math::Matrix<dax::Scalar,Size,Size> &matrix);

template<>
void NonSingularMatrix<1>(dax::math::Matrix<dax::Scalar,1,1> &matrix)
{
  matrix(0,0) = 1;
}

template<>
void NonSingularMatrix<2>(dax::math::Matrix<dax::Scalar,2,2> &matrix)
{
  matrix(0,0) = -5;  matrix(0,1) =  6;
  matrix(1,0) = -7;  matrix(1,1) = -2;
}

template<>
void NonSingularMatrix<3>(dax::math::Matrix<dax::Scalar,3,3> &matrix)
{
  matrix(0,0) =  1;  matrix(0,1) = -2;  matrix(0,2) =  3;
  matrix(1,0) =  6;  matrix(1,1) =  7;  matrix(1,2) = -1;
  matrix(2,0) = -3;  matrix(2,1) =  1;  matrix(2,2) =  4;
}

template<>
void NonSingularMatrix<4>(dax::math::Matrix<dax::Scalar,4,4> &matrix)
{
  matrix(0,0) =  2;  matrix(0,1) =  1;  matrix(0,2) =  0;  matrix(0,3) =  3;
  matrix(1,0) = -1;  matrix(1,1) =  0;  matrix(1,2) =  2;  matrix(1,3) =  4;
  matrix(2,0) =  4;  matrix(2,1) = -2;  matrix(2,2) =  7;  matrix(2,3) =  0;
  matrix(3,0) = -4;  matrix(3,1) =  3;  matrix(3,2) =  5;  matrix(3,3) =  1;
}

template<>
void NonSingularMatrix<5>(dax::math::Matrix<dax::Scalar,5,5> &mat)
{
  mat(0,0) = 2;  mat(0,1) = 1;  mat(0,2) = 3;  mat(0,3) = 7;  mat(0,4) = 5;
  mat(1,0) = 3;  mat(1,1) = 8;  mat(1,2) = 7;  mat(1,3) = 9;  mat(1,4) = 8;
  mat(2,0) = 3;  mat(2,1) = 4;  mat(2,2) = 1;  mat(2,3) = 6;  mat(2,4) = 2;
  mat(3,0) = 4;  mat(3,1) = 0;  mat(3,2) = 2;  mat(3,3) = 2;  mat(3,4) = 3;
  mat(4,0) = 7;  mat(4,1) = 9;  mat(4,2) = 1;  mat(4,3) = 5;  mat(4,4) = 4;
}

template<int Size>
void SingularMatrix(
    dax::math::Matrix<dax::Scalar,Size,Size> &singularMatrix)
{
  FOR_ROW_COL(singularMatrix)
    {
    singularMatrix(row,col) = row+col;
    }
  if (Size > 1)
    {
    dax::math::MatrixSetRow(singularMatrix,
                            0,
                            dax::math::MatrixRow(singularMatrix, (Size+1)/2));
    }
}

// A simple but slow implementation of finding a determinant for comparison
// purposes.
template<int Size>
dax::Scalar RecursiveDeterminant(
    const dax::math::Matrix<dax::Scalar,Size,Size> &A)
{
  dax::math::Matrix<dax::Scalar,Size-1,Size-1> cofactorMatrix;
  dax::Scalar sum = 0.0;
  dax::Scalar sign = 1.0;
  for (int rowIndex = 0; rowIndex < Size; rowIndex++)
    {
    // Create the cofactor matrix for entry A(rowIndex,0)
    for (int cofactorRowIndex = 0;
         cofactorRowIndex < rowIndex;
         cofactorRowIndex++)
      {
      for (int colIndex = 1; colIndex < Size; colIndex++)
        {
        cofactorMatrix(cofactorRowIndex,colIndex-1) =
            A(cofactorRowIndex,colIndex);
        }
      }
    for (int cofactorRowIndex = rowIndex+1;
         cofactorRowIndex < Size;
         cofactorRowIndex++)
      {
      for (int colIndex = 1; colIndex < Size; colIndex++)
        {
        cofactorMatrix(cofactorRowIndex-1,colIndex-1) =
            A(cofactorRowIndex,colIndex);
        }
      }
    sum += sign * A(rowIndex,0) * RecursiveDeterminant(cofactorMatrix);
    sign = -sign;
    }
  return sum;
}

template<>
dax::Scalar RecursiveDeterminant<1>(const dax::math::Matrix<dax::Scalar,1,1> &A)
{
  return A(0,0);
}

template<class MatrixType, int Size>
struct SquareMatrixTest {
  static const int SIZE = Size;

  static void CheckMatrixSize()
  {
    std::cout << "Check reported matrix size." << std::endl;
    DAX_TEST_ASSERT(MatrixType::NUM_ROWS == SIZE, "Matrix has wrong size.");
    DAX_TEST_ASSERT(MatrixType::NUM_COLUMNS == SIZE, "Matrix has wrong size.");
  }

  static void LUPFactor()
  {
    std::cout << "Test LUP-factorization" << std::endl;

    MatrixType A;
    NonSingularMatrix(A);
    const MatrixType originalMatrix = A;
    dax::Tuple<int,SIZE> permutationVector;
    dax::Scalar inversionParity;
    bool valid;

    dax::math::detail::MatrixLUPFactor(A,
                                       permutationVector,
                                       inversionParity,
                                       valid);
    DAX_TEST_ASSERT(valid, "Matrix declared singular?");

    // Reconstruct L and U matrices from A.
    MatrixType L(0);
    MatrixType U(0);
    FOR_ROW_COL(A)
      {
      if (row < col)
        {
        U(row,col) = A(row,col);
        }
      else //(row >= col)
        {
        L(row,col) = A(row,col);
        if (row == col) { U(row,col) = 1; }
        }
      }

    // Check parity of permutation.
    dax::Scalar computedParity = 1.0;
    for (int i = 0; i < SIZE; i++)
      {
      for (int j = i+1; j < SIZE; j++)
        {
        if (permutationVector[i] > permutationVector[j])
          {
          computedParity = -computedParity;
          }
        }
      }
    DAX_TEST_ASSERT(inversionParity == computedParity,
                    "Got bad inversion parity.");

    // Reconstruct permutation matrix P.
    MatrixType P(0);
    for (int index = 0; index < Size; index++)
      {
      P(index, permutationVector[index]) = 1;
      }

    // Check that PA = LU is actually correct.
    MatrixType permutedMatrix = dax::math::MatrixMultiply(P,originalMatrix);
    MatrixType productMatrix = dax::math::MatrixMultiply(L,U);
    DAX_TEST_ASSERT(test_equal(permutedMatrix, productMatrix),
                    "LUP-factorization gave inconsistent answer.");

    // Check that a singular matrix is identified.
    MatrixType singularMatrix;
    SingularMatrix(singularMatrix);
    dax::math::detail::MatrixLUPFactor(singularMatrix,
                                       permutationVector,
                                       inversionParity,
                                       valid);
    DAX_TEST_ASSERT(!valid, "Expected matrix to be declared singular.");
  }

  static void SolveLinearSystem()
  {
    std::cout << "Solve a linear system" << std::endl;

    MatrixType A;
    dax::Tuple<dax::Scalar,SIZE> b;
    NonSingularMatrix(A);
    for (int index = 0; index < SIZE; index++)
      {
      b[index] = index+1;
      }
    bool valid;

    dax::Tuple<dax::Scalar,SIZE> x = dax::math::SolveLinearSystem(A, b, valid);
    DAX_TEST_ASSERT(valid, "Matrix declared singular?");

    // Check result.
    dax::Tuple<dax::Scalar,SIZE> check = dax::math::MatrixMultiply(A,x);
    DAX_TEST_ASSERT(test_equal(b, check),
                    "Linear solution does not solve equation.");

    // Check that a singular matrix is identified.
    MatrixType singularMatrix;
    SingularMatrix(singularMatrix);
    dax::math::SolveLinearSystem(singularMatrix, b, valid);
    DAX_TEST_ASSERT(!valid, "Expected matrix to be declared singular.");
  }

  static void Invert()
  {
    std::cout << "Invert a matrix." << std::endl;

    MatrixType A;
    NonSingularMatrix(A);
    bool valid;

    dax::math::Matrix<dax::Scalar,SIZE,SIZE> inverse =
        dax::math::MatrixInverse(A, valid);
    DAX_TEST_ASSERT(valid, "Matrix declared singular?");

    // Check result.
    dax::math::Matrix<dax::Scalar,SIZE,SIZE> product =
        dax::math::MatrixMultiply(A, inverse);
    DAX_TEST_ASSERT(
          test_equal(product, dax::math::MatrixIdentity<dax::Scalar,SIZE>()),
          "Matrix inverse did not give identity.");

    // Check that a singular matrix is identified.
    MatrixType singularMatrix;
    SingularMatrix(singularMatrix);
    dax::math::MatrixInverse(singularMatrix, valid);
    DAX_TEST_ASSERT(!valid, "Expected matrix to be declared singular.");
  }

  static void Determinant()
  {
    std::cout << "Compute a determinant." << std::endl;

    MatrixType A;
    NonSingularMatrix(A);

    dax::Scalar determinant = dax::math::MatrixDeterminant(A);

    // Check result.
    dax::Scalar determinantCheck = RecursiveDeterminant(A);
    DAX_TEST_ASSERT(test_equal(determinant, determinantCheck),
                    "Determinant computations do not agree.");

    // Check that a singular matrix has a zero determinant.
    MatrixType singularMatrix;
    SingularMatrix(singularMatrix);
    determinant = dax::math::MatrixDeterminant(singularMatrix);
    DAX_TEST_ASSERT(test_equal(determinant, dax::Scalar(0.0)),
                    "Non-zero determinant for singular matrix.");
  }

  static void Run()
  {
    std::cout << "-- " << SIZE << " x " << SIZE << std::endl;

    CheckMatrixSize();
    LUPFactor();
    SolveLinearSystem();
    Invert();
    Determinant();
  }

private:
  SquareMatrixTest();  // Not implemented
};

template<int Size>
void RunSquareMatrixTest()
{
  SquareMatrixTest<dax::math::Matrix<dax::Scalar,Size,Size>,Size>::Run();
}

template<class MatrixType, int Size>
void RunKnownSquareMatrixTest()
{
  SquareMatrixTest<MatrixType, Size>::Run();
}

struct MatrixTestFunctor
{
  template<typename T> void operator()(const T&) const {
    MatrixTest1<T,1>();
    MatrixTest1<T,2>();
    MatrixTest1<T,3>();
    MatrixTest1<T,4>();
    MatrixTest1<T,5>();
  }
};

struct VectorMultFunctor
{
  template<class VectorType>
  void operator()(const VectorType &) const {
    // This is mostly to make sure the compile can convert from Tuples
    // to vectors.
    const int SIZE = dax::VectorTraits<VectorType>::NUM_COMPONENTS;
    typedef typename dax::VectorTraits<VectorType>::ComponentType ComponentType;

    dax::math::Matrix<ComponentType,SIZE,SIZE> matrix(0);
    VectorType inVec;
    VectorType outVec;
    for (int index = 0; index < SIZE; index++)
      {
      matrix(index,index) = 1;
      inVec[index] = index+1;
      }

    outVec = dax::math::MatrixMultiply(matrix,inVec);
    DAX_TEST_ASSERT(test_equal(inVec, outVec), "Bad identity multiply.");

    outVec = dax::math::MatrixMultiply(inVec,matrix);
    DAX_TEST_ASSERT(test_equal(inVec, outVec), "Bad identity multiply.");
  }
};

void TestMatrices()
{
  std::cout << "****** Rectangle tests" << std::endl;
  dax::testing::Testing::TryAllTypes(
        MatrixTestFunctor(), dax::testing::Testing::TypeCheckScalar());

  std::cout << "****** Square tests" << std::endl;
  RunSquareMatrixTest<1>();
  RunSquareMatrixTest<2>();
  RunSquareMatrixTest<3>();
  RunSquareMatrixTest<4>();
  RunSquareMatrixTest<5>();

  std::cout << "***** Common square types" << std::endl;
  RunKnownSquareMatrixTest<dax::math::Matrix2x2,2>();
  RunKnownSquareMatrixTest<dax::math::Matrix3x3,3>();
  RunKnownSquareMatrixTest<dax::math::Matrix4x4,4>();

  std::cout << "***** Vector multiply tests" << std::endl;
  dax::testing::Testing::TryAllTypes(
        VectorMultFunctor(), dax::testing::Testing::TypeCheckVector());
}

} // anonymous namespace

int UnitTestMathMatrix(int, char *[])
{
  return dax::testing::Testing::Run(TestMatrices);
}
