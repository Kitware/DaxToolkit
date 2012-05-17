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

#include <dax/Types.h>
#include <dax/exec/WorkMapField.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <Worklets/Testing/CellMapError.worklet>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class CellType, class ExecAdapter>
struct CellMapErrorParameters
{
  typename CellType::template GridStructures<ExecAdapter>::TopologyType grid;
};

template<class CellType, class ExecAdapter>
struct CellMapError
{
  DAX_EXEC_EXPORT void operator()(
      CellMapErrorParameters<CellType, ExecAdapter> &parameters,
      dax::Id index,
      typename ExecAdapter::ErrorHandler &errorHandler)
  {
    dax::exec::WorkMapCell<CellType, ExecAdapter>
        work(parameters.grid, index, errorHandler);
    dax::worklet::testing::CellMapError(work);
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
         template <typename> class Container,
         class DeviceAdapter>
inline void CellMapError(const GridType &grid)
{
  typedef typename DeviceAdapter::template ExecutionAdapter<Container>
      ExecAdapter;

  typedef typename GridType::ExecutionTopologyStruct ExecutionTopologyType;
  ExecutionTopologyType execTopology
      = dax::cont::internal::ExecutionPackageGrid::GetExecutionObject(grid);

  typedef typename GridType::CellType CellType;

  typedef dax::exec::internal::kernel
      ::CellMapErrorParameters<CellType, ExecAdapter> Parameters;

  Parameters parameters;
  parameters.grid = execTopology;

  DeviceAdapter::Schedule(
        dax::exec::internal::kernel::CellMapError<CellType, ExecAdapter>(),
        parameters,
        grid.GetNumberOfCells(),
        ExecAdapter());
}

}
}
}
} //dax::cont::worklet::testing

#endif //__dax_cont_worklet_testing_CellMapError_h
