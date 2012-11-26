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
#ifndef __dax_exec_CellVertices_h
#define __dax_exec_CellVertices_h

#include <dax/CellTraits.h>
#include <dax/Types.h>

namespace dax {
namespace exec {

/// \brief Holds the point indices for a cell of a particular type.
///
/// This class is really is a convienience wrapper around a dax::Tuple.
///
template<class CellTag>
class CellVertices
{
public:
  const static int NUM_VERTICES = dax::CellTraits<CellTag>::NUM_VERTICES;
  typedef dax::Tuple<dax::Id, NUM_VERTICES> PointIndicesType;

  DAX_EXEC_EXPORT
  CellVertices() {  }

  DAX_EXEC_EXPORT
  CellVertices(const PointIndicesType &pointIndices)
    : PointIndices(pointIndices) {  }

  // Although this copy constructor should be identical to the default copy
  // constructor, we have noticed that NVCC's default copy constructor can
  // incur a significant slowdown.
  DAX_EXEC_EXPORT
  CellVertices(const CellVertices &src)
    : PointIndices(src.PointIndices) {  }

  DAX_EXEC_EXPORT
  dax::Id GetPointIndex(int vertexIndex) const
  {
    return this->PointIndices[vertexIndex];
  }

  DAX_EXEC_EXPORT
  void SetPointIndex(int vertexIndex, dax::Id pointIndex)
  {
    this->PointIndices[vertexIndex] = pointIndex;
  }

  DAX_EXEC_EXPORT
  const PointIndicesType &GetPointIndices() const { return this->PointIndices; }

  DAX_EXEC_EXPORT
  void SetPointIndices(const PointIndicesType pointIndices)
  {
    this->PointIndices = pointIndices;
  }

private:
  PointIndicesType PointIndices;
};

}
} // namespace dax::exec

#endif //__dax_exec_CellVertices_h
