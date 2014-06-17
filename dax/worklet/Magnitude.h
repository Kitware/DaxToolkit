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
#ifndef __Magnitude_worklet_
#define __Magnitude_worklet_

#include <dax/exec/WorkletMapField.h>

#include <dax/math/VectorAnalysis.h>

namespace dax {
namespace worklet {

class Magnitude : public dax::exec::WorkletMapField
{
public:
  typedef void ControlSignature(FieldIn, FieldOut);
  typedef void ExecutionSignature(_1,_2);

  DAX_EXEC_EXPORT
  void operator()(const dax::Vector3 &inValue,
                  dax::Scalar &outValue) const
  {
    outValue = dax::math::Magnitude(inValue);
  }
};

}
}

#endif
