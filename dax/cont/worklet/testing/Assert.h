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
#include <dax/internal/DataArray.h>
#include <dax/internal/GridTopologys.h>
#include <dax/exec/WorkMapField.h>
#include <dax/exec/internal/ErrorHandler.h>
#include <dax/exec/internal/FieldBuild.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <Worklets/Testing/Assert.worklet>

namespace dax {
namespace exec {
namespace kernel {

template<class CellType>
struct AssertParameters
{
  typename CellType::TopologyType grid;
};

template<class CellType>
struct Assert
{
  DAX_EXEC_EXPORT void operator()(
      AssertParameters<CellType> &parameters,
      dax::Id index,
      const dax::exec::internal::ErrorHandler &errorHandler)
  {
    dax::exec::WorkMapField<CellType> work(parameters.grid, errorHandler);
    work.SetIndex(index);
    dax::worklet::testing::Assert(work);
  }
};

}
}
} // dax::exec::kernel

namespace dax {
namespace cont {
namespace worklet {
namespace testing {

template<class GridType, class DeviceAdapter>
inline void Assert(const GridType &grid)
{
  typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
  GridPackageType gridPackage(grid);

  typedef typename GridPackageType::ExecutionCellType CellType;

  typedef dax::exec::kernel::AssertParameters<CellType> Parameters;
  Parameters parameters = {
    gridPackage.GetExecutionObject()
  };

  DeviceAdapter::Schedule(
        dax::exec::kernel::Assert<CellType>(),
        parameters,
        grid.GetNumberOfPoints());
}

}
}
}
} //dax::cont::worklet::testing

#endif //__dax_cont_worklet_testing_Assert_h
