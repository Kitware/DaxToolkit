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
#ifndef __dax_cont_worklet_testing_Assert_h
#define __dax_cont_worklet_testing_Assert_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/exec/WorkMapField.h>
#include <dax/exec/internal/ExecutionAdapter.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <Worklets/Testing/Assert.worklet>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class CellType, class ExecAdapter>
struct AssertParameters
{
  typename CellType::template GridStructures<ExecAdapter>::TopologyType grid;
};

template<class CellType, class ExecAdapter>
struct Assert
{
  DAX_EXEC_EXPORT void operator()(
      AssertParameters<CellType, ExecAdapter> &parameters,
      dax::Id index,
      const ExecAdapter &execAdapter)
  {
    dax::exec::WorkMapField<CellType, ExecAdapter>
        work(parameters.grid, index, execAdapter);
    dax::worklet::testing::Assert(work);
  }
};

}
}
}
} // dax::exec::internal::kernel

namespace dax {
namespace cont {
namespace worklet {
namespace testing {

template<class GridType,
         class Container,
         class Adapter>
inline void Assert(const GridType &grid)
{
  typedef dax::exec::internal::ExecutionAdapter<Container,Adapter> ExecAdapter;

  typedef typename GridType::ExecutionTopologyStruct ExecutionTopologyType;
  ExecutionTopologyType execTopology
      = dax::cont::internal::ExecutionPackageGrid(grid);

  typedef typename GridType::CellType CellType;

  typedef dax::exec::internal::kernel::AssertParameters<CellType, ExecAdapter>
      Parameters;

  Parameters parameters;
  parameters.grid = execTopology;

  dax::cont::internal::Schedule(
        dax::exec::internal::kernel::Assert<CellType, ExecAdapter>(),
        parameters,
        grid.GetNumberOfPoints(),
        Container(),
        Adapter());
}

}
}
}
} //dax::cont::worklet::testing

#endif //__dax_cont_worklet_testing_Assert_h
