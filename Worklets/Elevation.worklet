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
#ifndef __Elevation_worklet_
#define __Elevation_worklet_

#include <dax/exec/WorkletMapField.h>

#include <dax/math/Compare.h>
#include <dax/math/VectorAnalysis.h>

namespace dax {
namespace worklet {

class Elevation : public dax::exec::WorkletMapField
{
public:
  typedef void ControlSignature(Field(In), Field(Out));
  typedef void ExecutionSignature(_1,_2);


  DAX_CONT_EXPORT
  Elevation(const dax::Vector3 &lowPoint = dax::make_Vector3(0.0, 0.0, 0.0),
            const dax::Vector3 &highPoint = dax::make_Vector3(0.0, 0.0, 1.0),
            const dax::Vector2 &outputRange = dax::make_Vector2(0.0, 1.0))
    : LowPoint(lowPoint),
      LowOutput(outputRange[0]),
      OutputDistance(outputRange[1] - outputRange[0])
  {
    dax::Vector3 direction = highPoint-lowPoint;
    dax::Scalar length2 = dax::math::MagnitudeSquared(direction);
    this->ScaledDirection = (1/length2)*direction;
  }

  DAX_EXEC_EXPORT void operator()(const dax::Vector3 &inCoordinates,
                                  dax::Scalar &elevation) const
  {
    dax::Vector3 coordDirection = inCoordinates - this->LowPoint;
    dax::Scalar s = dax::dot(coordDirection, this->ScaledDirection);
    s = dax::math::Max(dax::Scalar(0), s);
    s = dax::math::Min(dax::Scalar(1), s);
    elevation = this->LowOutput + s*this->OutputDistance;
  }

private:
  dax::Vector3 LowPoint;
  dax::Vector3 ScaledDirection;
  dax::Scalar LowOutput;
  dax::Scalar OutputDistance;
};

}
}
#endif
