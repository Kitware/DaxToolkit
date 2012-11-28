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
#ifndef __dax_exec_CellField_h
#define __dax_exec_CellField_h

#include <dax/CellTraits.h>
#include <dax/Types.h>

namespace dax {
namespace exec {

/// \brief Holds field values for a cell of a particular type.
///
/// This class is really is a convienience wrapper around a dax::Tuple.
///
template<typename FieldType, class CellTag>
class CellField
{
public:
  const static int NUM_VERTICES = dax::CellTraits<CellTag>::NUM_VERTICES;
  typedef dax::Tuple<FieldType, NUM_VERTICES> ValuesTupleType;

  DAX_EXEC_EXPORT
  CellField() {  }

  DAX_EXEC_EXPORT
  CellField(const ValuesTupleType &values)
    : Values(values) {  }

  // Although this copy constructor should be identical to the default copy
  // constructor, we have noticed that NVCC's default copy constructor can
  // incur a significant slowdown.
  DAX_EXEC_EXPORT
  CellField(const CellField &src)
    : Values(src.Values) {  }

  DAX_EXEC_EXPORT
  FieldType GetValue(int vertexIndex) const
  {
    return this->Values[vertexIndex];
  }

  DAX_EXEC_EXPORT
  void SetValue(int vertexIndex, FieldType fieldValue)
  {
    this->Values[vertexIndex] = fieldValue;
  }

  DAX_EXEC_EXPORT
  const ValuesTupleType &GetValues() const { return this->Values; }

  DAX_EXEC_EXPORT
  void SetValues(const ValuesTupleType fieldValues)
  {
    this->Values = fieldValues;
  }

private:
  ValuesTupleType Values;
};

}
} // namespace dax::exec

#endif //__dax_exec_CellField_h
