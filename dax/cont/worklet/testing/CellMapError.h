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
#ifndef __dax_cont_worklet_testing_CellMapError_h
#define __dax_cont_worklet_testing_CellMapError_h

// TODO: This should be auto-generated.

#include <Worklets/Testing/CellMapError.worklet>

#include <dax/Types.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class TopologyType>
struct CellMapError
{
  typedef typename TopologyType::CellType CellType;

  DAX_CONT_EXPORT
  CellMapError(const dax::worklet::testing::CellMapError &worklet,
               const TopologyType &topology)
    : Worklet(worklet), Topology(topology) {  }

  DAX_EXEC_EXPORT void operator()(
      dax::Id cellIndex,
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->Worklet.SetErrorMessageBuffer(errorMessage);
    const dax::worklet::testing::CellMapError &constWorklet = this->Worklet;

    CellType cell(this->Topology, cellIndex);

    constWorklet(cell);
  }

private:
  dax::worklet::testing::CellMapError Worklet;
  const TopologyType &Topology;
};

}
}
}
} // dax::exec::internal::kernel

namespace dax {
namespace cont {
namespace worklet {
namespace testing {

// The arguments for this are a hack because there are no ArrayHandles passed
// to definitively define the device adapter. This is probably not a problem
// for pratical worklets. In the future we can change this to pass in a field
// that is not used.
template<class GridType,
         class DeviceAdapter>
inline void CellMapError(const GridType &grid, DeviceAdapter)
{
  dax::exec::internal::kernel::CellMapError<
      typename GridType::ExecutionTopologyStruct>
      kernel(dax::worklet::testing::CellMapError(),
             grid.PrepareForInput());

  dax::cont::internal::Schedule(kernel,
                                grid.GetNumberOfCells(),
                                DeviceAdapter());
}

}
}
}
} //dax::cont::worklet::testing

#endif //__dax_cont_worklet_testing_CellMapError_h
