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
#ifndef __CellMapError_worklet_
#define __CellMapError_worklet_

#include <dax/exec/WorkletMapCell.h>


namespace dax {
namespace worklet {
namespace testing {

class CellMapError : public dax::exec::WorkletMapCell
{
public:
  typedef void ControlSignature(TopologyIn);
  typedef void ExecutionSignature(_1);

  template<class CellTag>
  DAX_EXEC_EXPORT
  void operator()(CellTag) const
  {
    this->RaiseError("Testing execution error system.");
  }
};

}
}
}
#endif
