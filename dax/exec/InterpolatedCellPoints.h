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
#ifndef __dax_exec_InterpolatedCellPoints_h
#define __dax_exec_InterpolatedCellPoints_h

#include <dax/cont/internal/EdgeInterpolatedGrid.h>
#include <dax/exec/CellField.h>

namespace dax {
namespace exec {

/// \brief Holds the point indices for a cell of a particular type.
///
/// This class is really is a convenience wrapper around a dax::Tuple.
///
template<class CellTag>
class InterpolatedCellPoints : public dax::exec::CellField< dax::PointAsEdgeInterpolation, CellTag>
{
private:
  typedef dax::exec::CellField< dax::PointAsEdgeInterpolation, CellTag > Superclass;
public:
  const static int NUM_VERTICES = Superclass::NUM_VERTICES;
  typedef typename Superclass::TupleType TupleType;

  DAX_EXEC_CONT_EXPORT
  InterpolatedCellPoints() {  }

  DAX_EXEC_CONT_EXPORT
  InterpolatedCellPoints(const TupleType &pointIndices) : Superclass(pointIndices) {  }

  DAX_CONT_EXPORT
  InterpolatedCellPoints(dax::PointAsEdgeInterpolation value) : Superclass(value) {  }

  // Although this copy constructor should be identical to the default copy
  // constructor, we have noticed that NVCC's default copy constructor can
  // incur a significant slowdown.
  DAX_EXEC_CONT_EXPORT
  InterpolatedCellPoints(const InterpolatedCellPoints &src) : Superclass(src) {  }

  //Allow easier setting of the interpolation value.
  DAX_EXEC_EXPORT
  void SetInterpolationPoint( dax::Id index, dax::Id pos1, dax::Id pos2,
                              dax::Scalar weight )
    {

    dax::PointAsEdgeInterpolation interpolationInfo( pos1, pos2, weight);
    (*this)[index]=interpolationInfo;
    }
};

}
} // namespace dax::exec

namespace dax {

/// Implementation of VectorTraits for a InterpolatedCellPoints so that it can be treated
/// like a vector.
///
template<class CellTag>
struct VectorTraits<dax::exec::InterpolatedCellPoints<CellTag> >
{
  typedef dax::exec::InterpolatedCellPoints<CellTag> InterpolatedCellPointsType;
  typedef dax::Id ComponentType;
  static const int NUM_COMPONENTS = InterpolatedCellPointsType::NUM_VERTICES;
  typedef typename internal::VectorTraitsMultipleComponentChooser<
      NUM_COMPONENTS>::Type HasMultipleComponents;

  DAX_EXEC_EXPORT
  static const ComponentType &GetComponent(const InterpolatedCellPointsType &vector,
                                           int component) {
    return vector[component];
  }
  DAX_EXEC_EXPORT
  static ComponentType &GetComponent(InterpolatedCellPointsType &vector, int component) {
    return vector[component];
  }

  DAX_EXEC_EXPORT static void SetComponent(InterpolatedCellPointsType &vector,
                                           int component,
                                           ComponentType value) {
    vector[component] = value;
  }

  DAX_EXEC_CONT_EXPORT
  static dax::Tuple<ComponentType,NUM_COMPONENTS>
  ToTuple(const InterpolatedCellPointsType &vector)
  {
    return vector.GetAsTuple();
  }
};

/// Implementation of TypeTraits for a CellField.
///
template<class CellTag>
struct TypeTraits<dax::exec::InterpolatedCellPoints<CellTag> > {
  typedef dax::TypeTraitsIntegerTag NumericTag;
  typedef dax::TypeTraitsVectorTag DimensionalityTag;
};
template<>
struct TypeTraits<dax::exec::InterpolatedCellPoints<dax::CellTagVertex> > {
  typedef dax::TypeTraitsIntegerTag NumericTag;
  typedef dax::TypeTraitsScalarTag DimensionalityTag;
};

} // namespace dax

#endif //__dax_exec_InterpolatedCellPoints_h
