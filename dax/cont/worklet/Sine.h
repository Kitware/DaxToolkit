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
#ifndef __dax_cont_worklet_Sine_h
#define __dax_cont_worklet_Sine_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapField.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/ErrorControlBadValue.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <Worklets/Sine.worklet>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class CellType, typename FieldType, class ExecAdapter>
struct SineParameters
{
  typename CellType::TopologyType grid;
  dax::exec::FieldIn<FieldType, ExecAdapter> inField;
  dax::exec::FieldOut<FieldType, ExecAdapter> outField;
};

template<class CellType, typename FieldType, class ExecAdapter>
struct Sine
{
  DAX_EXEC_EXPORT void operator()(
      SineParameters<CellType, FieldType, ExecAdapter> &parameters,
      dax::Id index,
      typename ExecAdapter::ErrorHandler &errorHandler)
  {
    dax::exec::WorkMapField<CellType, ExecAdapter>
        work(parameters.grid, index, errorHandler);
    dax::worklet::Sine(work,
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

// Should be templated on grid type too.
template<class GridType,
         typename FieldType,
         template <typename> class Container,
         class DeviceAdapter>
DAX_CONT_EXPORT void Sine(
    const GridType &grid,
    const dax::cont::ArrayHandle<FieldType,Container,DeviceAdapter> &inHandle,
    dax::cont::ArrayHandle<FieldType,Container,DeviceAdapter> &outHandle)
{
  dax::Id fieldSize;
  if (inHandle.GetNumberOfValues() == grid.GetNumberOfPoints())
    {
    fieldSize = grid.GetNumberOfPoints();
    }
  else if (inHandle.GetNumberOfValues() == grid.GetNumberOfCells())
    {
    fieldSize = grid.GetNumberOfCells();
    }
  else
    {
    throw dax::cont::ErrorControlBadValue(
          "Number of array entries neither cells nor points.");
    }

  typedef typename DeviceAdapter::template ExecutionAdapter<Container>
      ExecAdapter;

  typedef typename GridType::ExecutionTopologyStruct ExecutionTopologyType;
  ExecutionTopologyType execTopology
      = dax::cont::internal::ExecutionPackageGrid::GetExecutionObject(grid);

  dax::exec::FieldPointIn<FieldType, ExecAdapter> fieldIn
      = dax::cont::internal::ExecutionPackageField
        ::GetExecutionObject<dax::exec::FieldPointIn>(inHandle, fieldSize);

  dax::exec::FieldCellOut<FieldType, ExecAdapter> fieldOut
      = dax::cont::internal::ExecutionPackageField
        ::GetExecutionObject<dax::exec::FieldCellOut>(outHandle, fieldSize);

  typedef typename GridType::CellType CellType;

  typedef dax::exec::internal::kernel::SineParameters<
      CellType, FieldType, ExecAdapter> Parameters;

  Parameters parameters;

  parameters.grid = execTopology;
  parameters.inField = fieldIn;
  parameters.outField = fieldOut;

  DeviceAdapter::Schedule(
        dax::exec::internal::kernel::Sine<CellType, FieldType, ExecAdapter>(),
        parameters,
        fieldSize,
        ExecAdapter());
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_Sine_h
