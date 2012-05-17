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
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapField.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <Worklets/Elevation.worklet>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class CellType, class ExecAdapter>
struct ElevationParameters
{
  typename CellType::template GridStructures<ExecAdapter>::TopologyType grid;
  dax::exec::FieldCoordinatesIn<ExecAdapter> inCoordinates;
  dax::exec::FieldPointOut<dax::Scalar, ExecAdapter> outField;
};

template<class CellType, class ExecAdapter>
struct Elevation
{
  DAX_EXEC_EXPORT void operator()(
      ElevationParameters<CellType, ExecAdapter> &parameters,
      dax::Id index,
      typename ExecAdapter::ErrorHandler &errorHandler)
  {
    dax::exec::WorkMapField<CellType,ExecAdapter>
        work(parameters.grid, index, errorHandler);
    dax::worklet::Elevation(work,
                            parameters.inCoordinates,
                            parameters.outField);
  }
};

}
}
}
} // dax::exec::internal::kernel

namespace dax {
namespace cont {
namespace worklet {

template<class GridType,
         template <typename> class Container,
         class DeviceAdapter>
DAX_CONT_EXPORT void Elevation(
    const GridType &grid,
    const typename GridType::PointCoordinatesType &points,
    dax::cont::ArrayHandle<dax::Scalar, Container, DeviceAdapter> &outHandle)
{
  typedef typename DeviceAdapter::template ExecutionAdapter<Container>
      ExecAdapter;

  typedef typename GridType::ExecutionTopologyStruct ExecutionTopologyType;
  ExecutionTopologyType execTopology
      = dax::cont::internal::ExecutionPackageGrid::GetExecutionObject(grid);

  dax::exec::FieldCoordinatesIn<ExecAdapter> fieldCoordinates
      = dax::cont::internal::ExecutionPackageField
        ::GetExecutionObject<dax::exec::FieldCoordinatesIn>(points, grid);

  dax::exec::FieldCellOut<dax::Vector3, ExecAdapter> fieldOut
      = dax::cont::internal::ExecutionPackageField
        ::GetExecutionObject<dax::exec::FieldCellOut>(outHandle, grid);

  typedef typename GridType::CellType CellType;

  typedef dax::exec::internal::kernel::ElevationParameters<CellType,ExecAdapter>
      Parameters;

  Parameters parameters;
  parameters.grid = execTopology;
  parameters.inCoordinates = fieldCoordinates;
  parameters.outField = fieldOut;

  DeviceAdapter::Schedule(
        dax::exec::internal::kernel::Elevation<CellType, ExecAdapter>(),
        parameters,
        grid.GetNumberOfPoints(),
        ExecAdapter());
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_Elevation_h
