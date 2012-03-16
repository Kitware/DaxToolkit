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

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>
#include <dax/cont/internal/ScheduleMapAdapter.h>

namespace dax {
namespace exec {
namespace kernel {
namespace internal {


template <typename CellType>
DAX_WORKLET void ExtractCoordinates(dax::exec::WorkMapField<CellType> work,
                                    dax::Id newIndex,
                                    const dax::exec::FieldCoordinates &pointCoord,
                                    dax::exec::Field<dax::Vector3> &output)
{
  dax::Vector3 pointLocation = work.GetFieldValue(pointCoord);
  dax::Vector3* location = output.GetArray().GetPointer();
  location[newIndex]=pointLocation;
}

template<class CellType>
struct ExtractCoordinatesParameters
{
  typename CellType::TopologyType grid;
  dax::exec::FieldCoordinates inCoordinates;
  dax::exec::Field<dax::Vector3> outField;
};

template<class CellType>
struct ExtractCoordinatesFunctor {
  DAX_EXEC_EXPORT void operator()(
      ExtractCoordinatesParameters<CellType> &parameters,
      dax::Id key, dax::Id value,
      const dax::exec::internal::ErrorHandler &errorHandler)
  {
    dax::exec::WorkMapField<CellType> work(parameters.grid, errorHandler);
    work.SetIndex(value);
    dax::exec::kernel::internal::ExtractCoordinates(work,
                                                    key,
                                                    parameters.inCoordinates,
                                                    parameters.outField);
  }
};

}
}
}
} //dax::exec::kernel::internal

namespace dax {
namespace cont {
namespace internal {

template<typename DeviceAdapter, typename GridType>
class ExtractCoordinates
{
public:
  /// Extract a subset of the cells Coordinates.
  ExtractCoordinates(const GridType& grid,
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


}
}
}
#endif // __dax_exec_internal_ExtractCoordinates_h
