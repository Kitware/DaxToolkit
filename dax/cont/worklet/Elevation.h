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
#include <dax/exec/internal/ExecutionAdapter.h>
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
      const ExecAdapter &execAdapter) const
  {
    dax::exec::WorkMapField<CellType,ExecAdapter>
        work(parameters.grid, index, execAdapter);
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
         class Container,
         class Adapter>
DAX_CONT_EXPORT void Elevation(
    const GridType &grid,
    const typename GridType::PointCoordinatesType &points,
    dax::cont::ArrayHandle<dax::Scalar, Container, Adapter> &outHandle)
{
  typedef dax::exec::internal::ExecutionAdapter<Container,Adapter> ExecAdapter;

  typedef typename GridType::ExecutionTopologyStruct ExecutionTopologyType;
  ExecutionTopologyType execTopology
      = dax::cont::internal::ExecutionPackageGrid(grid);

  dax::exec::FieldCoordinatesIn<ExecAdapter> fieldCoordinates =
      dax::cont::internal::ExecutionPackageField<dax::exec::FieldCoordinatesIn>(
        points, grid);

  dax::exec::FieldPointOut<dax::Scalar, ExecAdapter> fieldOut
      = dax::cont::internal::ExecutionPackageField<dax::exec::FieldPointOut>(
        outHandle, grid);

  typedef typename GridType::CellType CellType;

  typedef dax::exec::internal::kernel::ElevationParameters<CellType,ExecAdapter>
      Parameters;

  Parameters parameters;
  parameters.grid = execTopology;
  parameters.inCoordinates = fieldCoordinates;
  parameters.outField = fieldOut;

  dax::cont::internal::Schedule(
        dax::exec::internal::kernel::Elevation<CellType, ExecAdapter>(),
        parameters,
        grid.GetNumberOfPoints(),
        Container(),
        Adapter());
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_Elevation_h
