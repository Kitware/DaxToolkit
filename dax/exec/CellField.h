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
#include <dax/TypeTraits.h>
#include <dax/VectorTraits.h>

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
  typedef dax::Tuple<FieldType, NUM_VERTICES> TupleType;

  DAX_EXEC_CONT_EXPORT
  CellField() {  }

  DAX_EXEC_CONT_EXPORT
  CellField(const TupleType &values)
    : Values(values) {  }

  DAX_CONT_EXPORT
  CellField(const FieldType &value)
    : Values(value) {  }

  // Although this copy constructor should be identical to the default copy
  // constructor, we have noticed that NVCC's default copy constructor can
  // incur a significant slowdown.
  DAX_EXEC_CONT_EXPORT
  CellField(const CellField &src)
    : Values(src.Values) {  }

  DAX_EXEC_EXPORT
  const FieldType &operator[](int vertexIndex) const {
    return this->Values[vertexIndex];
  }

  DAX_EXEC_EXPORT
  FieldType &operator[](int vertexIndex) {
    return this->Values[vertexIndex];
  }

  DAX_EXEC_EXPORT
  const TupleType &GetAsTuple() const { return this->Values; }

  DAX_EXEC_EXPORT
  void SetFromTuple(const TupleType &fieldValues)
  {
    this->Values = fieldValues;
  }

private:
  TupleType Values;
};

}
} // namespace dax::exec

namespace dax {

/// Implementation of VectorTraits for a CellField so that it can be treated
/// like a vector.
///
template<typename FieldType, class CellTag>
struct VectorTraits<dax::exec::CellField<FieldType, CellTag> >
{
  typedef dax::exec::CellField<FieldType,CellTag> CellFieldType;
  typedef FieldType ComponentType;
  static const int NUM_COMPONENTS = CellFieldType::NUM_VERTICES;
  typedef typename internal::VectorTraitsMultipleComponentChooser<
      NUM_COMPONENTS>::Type HasMultipleComponents;

  DAX_EXEC_EXPORT
  static const ComponentType &GetComponent(const CellFieldType &vector,
                                           int component) {
    return vector[component];
  }
  DAX_EXEC_EXPORT
  static ComponentType &GetComponent(CellFieldType &vector, int component) {
    return vector[component];
  }

  DAX_EXEC_EXPORT static void SetComponent(CellFieldType &vector,
                                           int component,
                                           ComponentType value) {
    vector[component] = value;
  }

  DAX_EXEC_CONT_EXPORT
  static dax::Tuple<ComponentType,NUM_COMPONENTS>
  ToTuple(const CellFieldType &vector)
  {
    return vector.GetAsTuple();
  }
};

/// Implementation of TypeTraits for a CellField.
///
template<typename FieldType, class CellTag>
struct TypeTraits<dax::exec::CellField<FieldType, CellTag> > {
  typedef typename TypeTraits<FieldType>::NumericTag NumericTag;
  typedef dax::TypeTraitsVectorTag DimensionalityTag;
};
template<typename FieldType>
struct TypeTraits<dax::exec::CellField<FieldType, dax::CellTagVertex> > {
  typedef typename TypeTraits<FieldType>::NumericTag NumericTag;
  typedef dax::TypeTraitsScalarTag DimensionalityTag;
};

} // namespace dax

#endif //__dax_exec_CellField_h
