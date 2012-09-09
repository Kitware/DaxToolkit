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
#ifndef __Cosine_worklet_
#define __Cosine_worklet_

#include <dax/exec/WorkletMapField.h>
#include <dax/math/Trig.h>

namespace dax {
namespace worklet {

class Cosine : public dax::exec::WorkletMapField
{
public:
  typedef void ControlSignature(Field(In), Field(Out));
  typedef _2 ExecutionSignature(_1);


  template<class ValueType>
  DAX_EXEC_EXPORT
  ValueType operator()(const ValueType &inValue) const
  {
    return dax::math::Cos(inValue);
  }
};

}
}

#endif
