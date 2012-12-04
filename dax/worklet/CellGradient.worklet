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
#ifndef __CellGradient_worklet_
#define __CellGradient_worklet_

#include <dax/exec/CellField.h>
#include <dax/exec/Derivative.h>
#include <dax/exec/ParametricCoordinates.h>
#include <dax/exec/WorkletMapCell.h>


namespace dax {
namespace worklet {

class CellGradient : public dax::exec::WorkletMapCell
{
public:

  typedef void ControlSignature(Topology, Field(Point), Field(Point), Field(Out));
  typedef _4 ExecutionSignature(_1,_2,_3);

  template<class CellTag>
  DAX_EXEC_EXPORT
  dax::Vector3 operator()(
      CellTag cellTag,
      const dax::exec::CellField<dax::Vector3,CellTag> &coords,
      const dax::exec::CellField<dax::Scalar,CellTag> &pointField) const
  {
    dax::Vector3 parametricCellCenter =
        dax::exec::ParametricCoordinates<CellTag>::Center();
    return dax::exec::CellDerivative(parametricCellCenter,
                                     coords,
                                     pointField,
                                     cellTag);
  }
};

}
} // namespace dax::worklet

#endif
