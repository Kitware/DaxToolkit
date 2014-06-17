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
#ifndef __PointDataToCellData_worklet_
#define __PointDataToCellData_worklet_

#include <dax/exec/CellField.h>
#include <dax/exec/Interpolate.h>
#include <dax/exec/ParametricCoordinates.h>
#include <dax/exec/WorkletMapCell.h>

namespace dax {
namespace worklet {

class PointDataToCellData : public dax::exec::WorkletMapCell
{
public:

  typedef void ControlSignature(TopologyIn,FieldPointIn, FieldOut);
  typedef _3 ExecutionSignature(_2);

  template<class CellTag>
  DAX_EXEC_EXPORT
  dax::Scalar operator()(
      const dax::exec::CellField<dax::Scalar,CellTag> &pointField) const
  {
    dax::Vector3 center =  dax::exec::ParametricCoordinates<CellTag>::Center();
    return dax::exec::CellInterpolate(pointField,center,CellTag());
  }
};
}
}

#endif
