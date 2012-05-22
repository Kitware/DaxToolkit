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

#ifndef __dax_cont_internal_ExtractCoordinates_h
#define __dax_cont_internal_ExtractCoordinates_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapField.h>
#include <dax/exec/internal/ExecutionAdapter.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>
#include <dax/cont/internal/ScheduleMapAdapter.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {


template <typename CellType, class ExecAdapter>
DAX_WORKLET void ExtractCoordinates(
    const dax::exec::WorkMapField<CellType, ExecAdapter> &work,
    dax::Id newIndex,
    const dax::exec::FieldCoordinatesIn<ExecAdapter> &pointCoord,
    const dax::exec::FieldOut<dax::Vector3, ExecAdapter> &output)
{
  dax::Vector3 pointLocation = work.GetFieldValue(pointCoord);
  dax::Vector3* location = output.GetArray().GetPointer();
  location[newIndex]=pointLocation;
}

template<class TopologyType, class ExecutionAdapter>
struct ExtractCoordinatesParameters
{
  TopologyType grid;
  dax::exec::FieldCoordinatesIn<ExecutionAdapter> inCoordinates;
  dax::exec::FieldOut<dax::Vector3, ExecutionAdapter> outField;
};

template<class CellType, class ExecutionAdapter>
struct ExtractCoordinatesFunctor {
  DAX_EXEC_EXPORT void operator()(
      ExtractCoordinatesParameters<CellType, ExecutionAdapter> &parameters,
      dax::Id key, dax::Id value,
      const typename ExecutionAdapter::ErrorHandler &errorHandler)
  {
    dax::exec::WorkMapField<CellType, ExecutionAdapter>
        work(parameters.grid, value, errorHandler);
    dax::exec::internal::kernel::ExtractCoordinates(work,
                                                    key,
                                                    parameters.inCoordinates,
                                                    parameters.outField);
  }
};

}
}
}
} //dax::exec::internal::kernel

namespace dax {
namespace cont {
namespace internal {

template<
    class GridType,
    class ArrayContainerControlTag,
    class DeviceAdapterTag>
DAX_CONT_EXPORT
dax::cont::ArrayHandle<dax::Vector3, ArrayContainerControlTag, DeviceAdapterTag>
ExtractCoordinates(
    const GridType &grid,
    const dax::cont::ArrayHandle<dax::Id,
                                 ArrayContainerControlTag,
                                 DeviceAdapterTag> &extractIds)
{
  //Verify the input
  DAX_ASSERT_CONT(grid.GetNumberOfPoints() > 0);
  DAX_ASSERT_CONT(extractIds.GetNumberOfValues() > 0);
  DAX_ASSERT_CONT(extractIds.GetNumberOfValues() <= grid.GetNumberOfPoints());

  const dax::Id outputSize = extractIds.GetNumberOfValues();
  dax::cont::ArrayHandle<dax::Vector3,ArrayContainerControlTag,DeviceAdapterTag>
      coordinates;

  typedef dax::exec::internal
      ::ExecutionAdapter<ArrayContainerControlTag,DeviceAdapterTag> ExecAdapter;
  typedef typename GridType::ExecutionTopologyStruct TopologyType;
  typedef typename GridType::CellType CellType;
  typedef dax::exec::internal::kernel
      ::ExtractCoordinatesFunctor<CellType, ExecAdapter> FunctorType;

  dax::exec::internal::kernel::ExtractCoordinatesParameters<
      TopologyType,ExecAdapter> parameters;
  parameters.grid =
      dax::cont::internal::ExecutionPackageGrid(grid);
  parameters.inCoordinates =
      dax::cont::internal::ExecutionPackageField<dax::exec::FieldCoordinatesIn>(
        grid.GetPointCoordinates(), grid);
  parameters.outField =
      dax::cont::internal::ExecutionPackageField<dax::exec::FieldOut>(
        coordinates, outputSize);

  dax::cont::internal::ScheduleMap(FunctorType(), parameters, extractIds);
}

#if 0
template<typename DeviceAdapter, typename GridType>
class ExtractCoordinates
{
public:
  /// Extract a subset of the cells Coordinates.
  ExtractCoordinates(
      const GridType& grid,
      dax::cont::ArrayHandle<dax::Id,DeviceAdapter> &extractIds)
    {
    //verify the input
    DAX_ASSERT_CONT(grid.GetNumberOfPoints() > 0);
    DAX_ASSERT_CONT(extractIds.GetNumberOfEntries() > 0);
    DAX_ASSERT_CONT(extractIds.GetNumberOfEntries()<=grid.GetNumberOfPoints());
    this->DoExtract(grid,extractIds);
    this->Coordinates.CompleteAsOutput();
  }

  /// Returns an array handle to the execution enviornment data that
  /// contians the extracted Coordinates
  dax::cont::ArrayHandle<dax::Vector3,DeviceAdapter>& GetCoordinates()
    {return Coordinates;}

private:
  void DoExtract(const GridType& grid,
                 dax::cont::ArrayHandle<dax::Id,DeviceAdapter> &extractIds);

  dax::cont::ArrayHandle<dax::Vector3,DeviceAdapter> Coordinates;
};

//-----------------------------------------------------------------------------
template<typename DeviceAdapter, typename GridType>
inline void ExtractCoordinates<DeviceAdapter,GridType>::DoExtract(
    const GridType& grid,
    dax::cont::ArrayHandle<dax::Id,DeviceAdapter> &extractIds)
{
  typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
  typedef typename GridPackageType::ExecutionCellType CellType;

  //construct the input grid
  GridPackageType inPGrid(grid);

  dax::cont::internal::ExecutionPackageFieldCoordinatesInput
      <GridType, DeviceAdapter>
      fieldCoords(grid.GetPoints());

  //construct the Coordinates result array
  const dax::Id size(extractIds.GetNumberOfEntries());

  this->Coordinates = dax::cont::ArrayHandle<dax::Vector3,DeviceAdapter>(size);

  //we want the size of the points to be based on the numCells * points per cell
  dax::cont::internal::ExecutionPackageFieldOutput<dax::Vector3,DeviceAdapter>
      result(this->Coordinates, size);

  //construct the parameters list for the function
  dax::exec::kernel::internal::ExtractCoordinatesParameters<CellType> etParams =
                                      {
                                      inPGrid.GetExecutionObject(),
                                      fieldCoords.GetExecutionObject(),
                                      result.GetExecutionObject(),
                                      };

  dax::cont::internal::ScheduleMap(
        dax::exec::kernel::internal::ExtractCoordinatesFunctor<CellType>(),
        etParams,
        extractIds);

}
#endif


}
}
}
#endif // __dax_exec_internal_ExtractCoordinates_h
