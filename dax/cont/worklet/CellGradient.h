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
#ifndef __dax_cont_worklet_CellGradient_h
#define __dax_cont_worklet_CellGradient_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapCell.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <Worklets/CellGradient.worklet>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class CellType, class ExecAdapter>
struct CellGradientParameters
{
  typename CellType::template GridStructures<ExecAdapter>::TopologyType grid;
  dax::exec::FieldCoordinatesIn<ExecAdapter> inCoordinates;
  dax::exec::FieldPointIn<dax::Scalar, ExecAdapter> inField;
  dax::exec::FieldCellOut<dax::Vector3, ExecAdapter> outField;
};

template<class CellType, class ExecAdapter>
struct CellGradient {
  DAX_EXEC_EXPORT void operator()(
      CellGradientParameters<CellType, ExecAdapter> &parameters,
      dax::Id index,
      const ExecAdapter &execAdapter)
  {
    dax::exec::WorkMapCell<CellType, ExecAdapter> work(
          parameters.grid, index, execAdapter);
    dax::worklet::CellGradient(work,
                               parameters.inCoordinates,
                               parameters.inField,
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
         class DeviceAdapter>
DAX_CONT_EXPORT void CellGradient(
    const GridType &grid,
    const typename GridType::PointCoordinatesType &points,
    dax::cont::ArrayHandle<dax::Scalar, Container, DeviceAdapter> &inHandle,
    dax::cont::ArrayHandle<dax::Vector3, Container, DeviceAdapter> &outHandle)
{
  typedef dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter>
      ExecAdapter;

  typedef typename GridType::ExecutionTopologyStruct ExecutionTopologyType;
  ExecutionTopologyType execTopology
      = dax::cont::internal::ExecutionPackageGrid(grid);

  dax::exec::FieldCoordinatesIn<ExecAdapter> fieldCoordinates
      = dax::cont::internal
      ::ExecutionPackageField<dax::exec::FieldCoordinatesIn>(points, grid);

  dax::exec::FieldPointIn<dax::Scalar, ExecAdapter> fieldIn
      = dax::cont::internal::ExecutionPackageField<dax::exec::FieldPointIn>(
        inHandle, grid);

  dax::exec::FieldCellOut<dax::Vector3, ExecAdapter> fieldOut
      = dax::cont::internal::ExecutionPackageField<dax::exec::FieldCellOut>(
        outHandle, grid);

  typedef typename GridType::CellType CellType;

  typedef dax::exec::internal::kernel
      ::CellGradientParameters<CellType, ExecAdapter> Parameters;

  Parameters parameters;
  parameters.grid = execTopology;
  parameters.inCoordinates = fieldCoordinates;
  parameters.inField = fieldIn;
  parameters.outField = fieldOut;

  dax::cont::internal::Schedule(
        dax::exec::internal::kernel::CellGradient<CellType, ExecAdapter>(),
        parameters,
        grid.GetNumberOfCells(),
        Container(),
        DeviceAdapter());
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_CellGradient_h
