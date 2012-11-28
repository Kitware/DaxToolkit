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

#include <dax/exec/CellField.h>

namespace dax {
namespace exec {

/// \brief Holds the point indices for a cell of a particular type.
///
/// This class is really is a convienience wrapper around a dax::Tuple.
///
template<class CellTag>
class CellVertices : public dax::exec::CellField<dax::Id, CellTag>
{
private:
  typedef dax::exec::CellField<dax::Id, CellTag> Superclass;
public:
  const static int NUM_VERTICES = Superclass::NUM_VERTICES;
  typedef typename Superclass::TupleType TupleType;

  DAX_EXEC_EXPORT
  CellVertices() {  }

  DAX_EXEC_EXPORT
  CellVertices(const TupleType &pointIndices)
    : Superclass(pointIndices) {  }

  // Although this copy constructor should be identical to the default copy
  // constructor, we have noticed that NVCC's default copy constructor can
  // incur a significant slowdown.
  DAX_EXEC_EXPORT
  CellVertices(const CellVertices &src)
    : Superclass(src) {  }
};

}
} // namespace dax::exec

#endif //__dax_exec_CellVertices_h
