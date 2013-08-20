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
#ifndef __CellAverage_worklet_
#define __CellAverage_worklet_

#include <dax/exec/CellField.h>
#include <dax/exec/VectorOperations.h>
#include <dax/exec/WorkletMapCell.h>
#include <dax/exec/WorkletMapField.h>

namespace dax {
namespace worklet {

namespace{
struct Add
{
  DAX_EXEC_EXPORT
  dax::Scalar operator()(dax::Scalar a, dax::Scalar b)
  {
    return a+b;
  }
};
}

class CellAverage : public dax::exec::WorkletMapCell
{
public:

  typedef void ControlSignature(Topology, Field(Point), Field(Out));
  typedef _3 ExecutionSignature(_1,_2);

  template<class CellTag>
  DAX_EXEC_EXPORT
  dax::Scalar operator()(
    CellTag, const dax::exec::CellField<dax::Scalar,CellTag> &values) const
  {
    Add add;
    return (dax::exec::VectorReduce(values,add)/values.NUM_VERTICES);
  }
};
}
}
#endif
