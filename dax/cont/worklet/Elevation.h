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
#ifndef __dax_cont_worklet_Elevation_h
#define __dax_cont_worklet_Elevation_h

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

#include <Worklets/Elevation.worklet>

namespace dax {
namespace exec {
namespace kernel {

template<class CellType>
struct ElevationParameters
{
  typename CellType::TopologyType grid;
  dax::exec::FieldCoordinates inCoordinates;
  dax::exec::FieldPoint<dax::Scalar> outField;
};

template<class CellType>
struct Elevation
{
  DAX_EXEC_EXPORT void operator()(
      ElevationParameters<CellType> &parameters,
      dax::Id index,
      const dax::exec::internal::ErrorHandler &errorHandler)
  {
    dax::exec::WorkMapField<CellType> work(parameters.grid, errorHandler);
    work.SetIndex(index);
    dax::worklet::Elevation(work,
                            parameters.inCoordinates,
                            parameters.outField);
  }
};

}
}
} // dax::cuda::exec::kernel

namespace dax {
namespace cont {
namespace worklet {

template<class GridType, class DeviceAdapter>
inline void Elevation(
    const GridType &grid,
    const typename GridType::Points &points,
    dax::cont::ArrayHandle<dax::Scalar, DeviceAdapter> &outHandle)
{
  typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
  GridPackageType gridPackage(grid);

  dax::cont::internal::ExecutionPackageFieldCoordinatesInput
      <GridType, DeviceAdapter>
      fieldCoordinates(points);

  dax::cont::internal::ExecutionPackageFieldPointOutput
      <dax::Scalar, DeviceAdapter>
      outField(outHandle, grid);

  typedef typename GridPackageType::ExecutionCellType CellType;

  typedef dax::exec::kernel::ElevationParameters<CellType> Parameters;
  Parameters parameters = {
    gridPackage.GetExecutionObject(),
    fieldCoordinates.GetExecutionObject(),
    outField.GetExecutionObject()
  };

  DeviceAdapter::Schedule(
        dax::exec::kernel::Elevation<CellType>(),
        parameters,
        grid.GetNumberOfPoints());
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_Elevation_h
