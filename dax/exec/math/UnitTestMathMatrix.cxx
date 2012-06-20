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

#define FOR_ROW_COL \
  for(int row=0; row < NUM_ROWS; row++) for(int col=0; col < NUM_COLS; col++)

#define MAT_VALUE(row, col) (static_cast<T>((row) + 10*(col)))

template<typename T, int NumRow, int NumCol>
void MatrixTest()
{
  const int NUM_ROWS = NumRow;
  const int NUM_COLS = NumCol;
  typedef dax::exec::math::Matrix<T,NUM_ROWS,NUM_COLS> MatrixType;
  typedef dax::Tuple<T,NUM_ROWS> ColumnType;
  typedef dax::Tuple<T,NUM_COLS> RowType;

  std::cout << "-- " << NUM_ROWS << " x " << NUM_COLS << std::endl;

  std::cout << "Basic creation." << std::endl;
  MatrixType matrix(5);
  FOR_ROW_COL
    {
    DAX_TEST_ASSERT(matrix(row,col) == static_cast<T>(5),
                    "Constant set incorrect.");
    }

  std::cout << "Basic accessors." << std::endl;
  FOR_ROW_COL
    {
    matrix[row][col] = MAT_VALUE(row,col)*2;
    }
  FOR_ROW_COL
    {
    DAX_TEST_ASSERT(matrix(row,col) == MAT_VALUE(row,col)*2,
                    "Bad set or retreive.");
    const MatrixType const_matrix = matrix;
    DAX_TEST_ASSERT(const_matrix(row,col) == MAT_VALUE(row,col)*2,
                    "Bad set or retreive.");
    }

  FOR_ROW_COL
    {
    matrix(row,col) = MAT_VALUE(row,col);
    }
  const MatrixType const_matrix = matrix;
  FOR_ROW_COL
    {
    DAX_TEST_ASSERT(matrix[row][col] == MAT_VALUE(row,col),
                    "Bad set or retreive.");
    DAX_TEST_ASSERT(const_matrix[row][col] == MAT_VALUE(row,col),
                    "Bad set or retreive.");
    }
  DAX_TEST_ASSERT(matrix == const_matrix,
                  "Equal test operator not working.");
  DAX_TEST_ASSERT(!(matrix != const_matrix),
                  "Not-Equal test operator not working.");
  DAX_TEST_ASSERT(test_equal(matrix, const_matrix),
                  "Vector-based equal test not working.");

  std::cout << "Access by row or column" << std::endl;
  FOR_ROW_COL
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
  FOR_ROW_COL
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
  FOR_ROW_COL
    {
    DAX_TEST_ASSERT(matrix(row,NUM_COLS-col-1) == const_matrix(row,col),
                    "Columns not set right.");
    }

  std::cout << "Try multiply." << std::endl;
    {
    dax::exec::math::Matrix<T,NUM_COLS,4> rightFactor;
    for (int index = 0; index < NUM_COLS*4; index++)
      {
      dax::VectorTraits<dax::exec::math::Matrix<T,NUM_COLS,4> >::SetComponent(
            rightFactor, index, index);
      }
    dax::exec::math::Matrix<T,NUM_ROWS,4> product
        = dax::exec::math::MatrixMultiply(const_matrix, rightFactor);
    for (int rowIndex = 0; rowIndex < NUM_ROWS; rowIndex++)
      {
      for (int colIndex = 0; colIndex < 4; colIndex++)
        {
        dax::Tuple<T,NUM_COLS> leftVector
            = dax::exec::math::MatrixRow(const_matrix, rowIndex);
        dax::Tuple<T,NUM_COLS> rightVector
            = dax::exec::math::MatrixColumn(rightFactor, colIndex);
        DAX_TEST_ASSERT(test_equal(product(rowIndex,colIndex),
                                   dax::dot(leftVector,rightVector)),
                        "Matrix multiple wrong.");
        }
      }
    }

  std::cout << "Check identity" << std::endl;
  matrix = dax::exec::math::MatrixMultiply(
        const_matrix, dax::exec::math::MatrixIdentity<T,NUM_COLS>());
  DAX_TEST_ASSERT(test_equal(matrix, const_matrix),
                  "Identity is not really identity.");

  std::cout << "Check transpose" << std::endl;
    {
    dax::exec::math::Matrix<T,NUM_COLS,NUM_ROWS> trans =
        dax::exec::math::MatrixTranspose(const_matrix);
    FOR_ROW_COL
      {
      DAX_TEST_ASSERT(const_matrix(row,col) == trans(col,row),
                      "Transpose wrong.");
      }
    }
}

template<typename T, int NumRow>
void MatrixTest1()
{
  MatrixTest<T,NumRow,1>();
  MatrixTest<T,NumRow,2>();
  MatrixTest<T,NumRow,3>();
  MatrixTest<T,NumRow,4>();
  MatrixTest<T,NumRow,5>();
}

template<typename T, int Size>
void SquareMatrixTest()
{
  const int SIZE = Size;
  typedef dax::exec::math::Matrix<T,SIZE,SIZE> MatrixType;

  std::cout << "-- " << SIZE << " x " << SIZE << std::endl;
}

struct MatrixTestFunctor
{
  template<typename T> void operator()(const T&) const {
    std::cout << "--- Rectangle tests" << std::endl;
    MatrixTest1<T,1>();
    MatrixTest1<T,2>();
    MatrixTest1<T,3>();
    MatrixTest1<T,4>();
    MatrixTest1<T,5>();

    std::cout << "--- Square tests" << std::endl;
    SquareMatrixTest<T,1>();
    SquareMatrixTest<T,2>();
    SquareMatrixTest<T,3>();
    SquareMatrixTest<T,4>();
    SquareMatrixTest<T,5>();
  }
};

void TestMatrices()
{
  dax::internal::Testing::TryAllTypes(
        MatrixTestFunctor(), dax::internal::Testing::TypeCheckScalar());
}

} // anonymous namespace

int UnitTestMathMatrix(int, char *[])
{
  return dax::internal::Testing::Run(TestMatrices);
}
