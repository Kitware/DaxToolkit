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
const dax::Tuple<T, NumCol> &MatrixRow(
    const dax::exec::math::Matrix<T,NumRow,NumCol> &matrix, int rowIndex)
{
  return matrix[rowIndex];
}

/// Returns a tuple containing the given column (indexed from 0) of the given
/// matrix.  Might not be as efficient as the Row function.
///
template<typename T, int NumRow, int NumCol>
dax::Tuple<T, NumRow> MatrixColumn(
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
void MatrixSetRow(dax::exec::math::Matrix<T,NumRow,NumCol> &matrix,
                  int rowIndex,
                  dax::Tuple<T,NumCol> rowValues)
{
  matrix[rowIndex] = rowValues;
}

/// Convenience function for setting a column of a matrix.
///
template<typename T, int NumRow, int NumCol>
void MatrixSetColumn(dax::exec::math::Matrix<T,NumRow,NumCol> &matrix,
                     int columnIndex,
                     dax::Tuple<T,NumRow> columnValues)
{
  for (int rowIndex = 0; rowIndex < NumRow; rowIndex++)
    {
    matrix(rowIndex, columnIndex) = columnValues[rowIndex];
    }
}

}
}
} // namespace dax::exec::math

#endif //__dax_exec_math_Matrix_h
