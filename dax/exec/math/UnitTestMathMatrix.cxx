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

#include <dax/exec/math/Matrix.h>

#include <dax/VectorTraits.h>

#include <dax/internal/Testing.h>

namespace {

#define FOR_ROW_COL(matrix) \
  for(int row=0; row < (matrix).NUM_ROWS; row++) \
  for(int col=0; col < (matrix).NUM_COLUMNS; col++)

template<typename T, int NumRow, int NumCol>
struct MatrixTest
{
  static const int NUM_ROWS = NumRow;
  static const int NUM_COLS = NumCol;
  typedef dax::exec::math::Matrix<T,NUM_ROWS,NUM_COLS> MatrixType;

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
      RowType rowvec = dax::exec::math::MatrixRow(const_matrix, row);
      DAX_TEST_ASSERT(rowvec[col] == const_matrix(row,col), "Bad get row.");
      ColumnType columnvec = dax::exec::math::MatrixColumn(const_matrix, col);
      DAX_TEST_ASSERT(columnvec[row] == const_matrix(row,col), "Bad get col.");
      }

    std::cout << "Set a row." << std::endl;
    for (int row = 0; row < NUM_ROWS; row++)
      {
      RowType rowvec = dax::exec::math::MatrixRow(const_matrix, NUM_ROWS-row-1);
      dax::exec::math::MatrixSetRow(matrix, row, rowvec);
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
          dax::exec::math::MatrixColumn(const_matrix, NUM_COLS-col-1);
      dax::exec::math::MatrixSetColumn(matrix, col, colvec);
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
    dax::exec::math::Matrix<T,NUM_COLS,4> rightFactor;
    for (int index = 0; index < NUM_COLS*4; index++)
      {
      dax::VectorTraits<dax::exec::math::Matrix<T,NUM_COLS,4> >::SetComponent(
            rightFactor, index, index);
      }

    dax::exec::math::Matrix<T,NUM_ROWS,4> product
        = dax::exec::math::MatrixMultiply(leftFactor, rightFactor);

    FOR_ROW_COL(product)
      {
      dax::Tuple<T,NUM_COLS> leftVector
          = dax::exec::math::MatrixRow(leftFactor, row);
      dax::Tuple<T,NUM_COLS> rightVector
          = dax::exec::math::MatrixColumn(rightFactor, col);
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
        dax::exec::math::MatrixMultiply(leftVector, matrixFactor);
    for (int index = 0; index < NUM_COLS; index++)
      {
      DAX_TEST_ASSERT(test_equal(leftResult[index], T(NUM_ROWS*(NUM_ROWS+1))),
                      "Vector/matrix multiple wrong.");
      }

    dax::Tuple<T,NUM_ROWS> rightResult =
        dax::exec::math::MatrixMultiply(matrixFactor, rightVector);
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

    dax::exec::math::Matrix<T,NUM_COLS,NUM_COLS> identityMatrix;
    dax::exec::math::MatrixIdentity(identityMatrix);

    MatrixType multMatrix =
        dax::exec::math::MatrixMultiply(originalMatrix, identityMatrix);

    DAX_TEST_ASSERT(test_equal(originalMatrix, multMatrix),
                    "Identity is not really identity.");

  }

  static void Transpose()
  {
    std::cout << "Check transpose" << std::endl;

    MatrixType originalMatrix = CreateMatrix();

    dax::exec::math::Matrix<T,NUM_COLS,NUM_ROWS> transMatrix =
        dax::exec::math::MatrixTranspose(originalMatrix);
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
struct SquareMatrixTest {
  static const int SIZE = Size;
  typedef dax::exec::math::Matrix<dax::Scalar,SIZE,SIZE> MatrixType;

  static void LUPFactor()
  {
    MatrixType A;
    FOR_ROW_COL(A)
      {
      A(row,col) = 2*col*col-((row+col)*row)+3;
      }
    const MatrixType originalMatrix = A;
    dax::Tuple<int,SIZE> permutationVector;
    bool valid;

    dax::exec::math::detail::MatrixLUPFactor(A, permutationVector, valid);
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

    // Reconstruct permutation matrix P.
    MatrixType P(0);
    for (int index = 0; index < Size; index++)
      {
      P(index, permutationVector[index]) = 1;
      }

    // Check that PA = LU is actually correct.
    MatrixType permutedMatrix =
        dax::exec::math::MatrixMultiply(P,originalMatrix);
    MatrixType productMatrix = dax::exec::math::MatrixMultiply(L,U);
    DAX_TEST_ASSERT(test_equal(permutedMatrix, productMatrix),
                    "LUP-factorization gave inconsistent answer.");

    // Check that a singular matrix is identified.
    MatrixType singularMatrix;
    FOR_ROW_COL(singularMatrix)
      {
      singularMatrix(row,col) = row+col;
      }
    if (Size > 1)
      {
      dax::exec::math::MatrixSetRow(singularMatrix,
                                    0,
                                    dax::exec::math::MatrixRow(singularMatrix,
                                                               (Size+1)/2));
      }
    dax::exec::math::detail::MatrixLUPFactor(singularMatrix,
                                             permutationVector,
                                             valid);
    DAX_TEST_ASSERT(!valid, "Expected matrix to be declared singular.");
  }

  static void SolveLinearSystem()
  {
    MatrixType A;
    dax::Tuple<dax::Scalar,SIZE> b;
    FOR_ROW_COL(A)
      {
      A(row,col) = 2*col*col-((row+col)*row)+3;
      b[row] = row+1;
      }
    bool valid;

    dax::Tuple<dax::Scalar,SIZE> x =
        dax::exec::math::SolveLinearSystem(A, b, valid);
    DAX_TEST_ASSERT(valid, "Matrix declared singular?");

    // Check result.
    dax::Tuple<dax::Scalar,SIZE> check =
        dax::exec::math::MatrixMultiply(A,x);
    DAX_TEST_ASSERT(test_equal(b, check),
                    "Linear solution does not solve equation.");

    // Check that a singular matrix is identified.
    MatrixType singularMatrix;
    FOR_ROW_COL(singularMatrix)
      {
      singularMatrix(row,col) = row+col;
      }
    if (Size > 1)
      {
      dax::exec::math::MatrixSetRow(singularMatrix,
                                    0,
                                    dax::exec::math::MatrixRow(singularMatrix,
                                                               (Size+1)/2));
      }
    dax::exec::math::SolveLinearSystem(singularMatrix, b, valid);
    DAX_TEST_ASSERT(!valid, "Expected matrix to be declared singular.");
  }

  static void Run()
  {
    std::cout << "-- " << SIZE << " x " << SIZE << std::endl;

    LUPFactor();
    SolveLinearSystem();
  }

private:
  SquareMatrixTest();  // Not implemented
};

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

    dax::exec::math::Matrix<ComponentType,SIZE,SIZE> matrix(0);
    VectorType inVec;
    VectorType outVec;
    for (int index = 0; index < SIZE; index++)
      {
      matrix(index,index) = 1;
      inVec[index] = index+1;
      }

    outVec = dax::exec::math::MatrixMultiply(matrix,inVec);
    DAX_TEST_ASSERT(test_equal(inVec, outVec), "Bad identity multiply.");

    outVec = dax::exec::math::MatrixMultiply(inVec,matrix);
    DAX_TEST_ASSERT(test_equal(inVec, outVec), "Bad identity multiply.");
  }
};

void TestMatrices()
{
  std::cout << "****** Rectangle tests" << std::endl;
  dax::internal::Testing::TryAllTypes(
        MatrixTestFunctor(), dax::internal::Testing::TypeCheckScalar());

  std::cout << "****** Square tests" << std::endl;
  SquareMatrixTest<1>::Run();
  SquareMatrixTest<2>::Run();
  SquareMatrixTest<3>::Run();
  SquareMatrixTest<4>::Run();
  SquareMatrixTest<5>::Run();

  std::cout << "***** Vector multiply tests" << std::endl;
  dax::internal::Testing::TryAllTypes(
        VectorMultFunctor(), dax::internal::Testing::TypeCheckVector());
}

} // anonymous namespace

int UnitTestMathMatrix(int, char *[])
{
  return dax::internal::Testing::Run(TestMatrices);
}
