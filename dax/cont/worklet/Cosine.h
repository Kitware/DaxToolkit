/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_worklet_Cosine_h
#define __dax_cont_worklet_Cosine_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/internal/DataArray.h>
#include <dax/internal/GridTopologys.h>
#include <dax/exec/WorkMapField.h>
#include <dax/exec/internal/ErrorHandler.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/ErrorControlBadValue.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <Worklets/Cosine.worklet>

namespace dax {
namespace exec {
namespace kernel {

template<class CellType, typename FieldType>
struct CosineParameters
{
  typename CellType::TopologyType grid;
  dax::exec::Field<FieldType> inField;
  dax::exec::Field<FieldType> outField;
};

template<class CellType, typename FieldType>
struct Cosine
{
  DAX_EXEC_EXPORT void operator()(
      CosineParameters<CellType, FieldType> &parameters,
      dax::Id index,
      const dax::exec::internal::ErrorHandler &errorHandler)
  {
    dax::exec::WorkMapField<CellType> work(parameters.grid, errorHandler);
    work.SetIndex(index);
    dax::worklet::Cosine(work,
                         parameters.inField,
                         parameters.outField);
  }
};

}
}
} // dax::exec::kernel

namespace dax {
namespace cont {
namespace worklet {

template<class GridType, typename FieldType, class DeviceAdapter>
inline void Cosine(const GridType &grid,
                   dax::cont::ArrayHandle<FieldType, DeviceAdapter> &inHandle,
                   dax::cont::ArrayHandle<FieldType, DeviceAdapter> &outHandle)
{
  dax::Id fieldSize;
  if (inHandle.GetNumberOfEntries() == grid.GetNumberOfPoints())
    {
    fieldSize = grid.GetNumberOfPoints();
    }
  else if (inHandle.GetNumberOfEntries() == grid.GetNumberOfCells())
    {
    fieldSize = grid.GetNumberOfCells();
    }
  else
    {
    throw dax::cont::ErrorControlBadValue(
          "Number of array entries neither cells nor points.");
    }

  typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
  GridPackageType gridPackage(grid);

  dax::cont::internal::ExecutionPackageFieldInput<FieldType, DeviceAdapter>
      inField(inHandle, fieldSize);

  dax::cont::internal::ExecutionPackageFieldOutput<FieldType, DeviceAdapter>
      outField(outHandle, fieldSize);

  typedef typename GridPackageType::ExecutionCellType CellType;

  typedef dax::exec::kernel::CosineParameters<CellType, FieldType> Parameters;
  Parameters parameters = {
    gridPackage.GetExecutionObject(),
    inField.GetExecutionObject(),
    outField.GetExecutionObject()
  };

  DeviceAdapter::Schedule(
        dax::exec::kernel::Cosine<CellType,FieldType>(),
        parameters,
        fieldSize);
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_Cosine_h
