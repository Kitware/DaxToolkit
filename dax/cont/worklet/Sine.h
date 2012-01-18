/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_worklet_Sine_h
#define __dax_cont_worklet_Sine_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>
#include <dax/exec/WorkMapField.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <Worklets/Sine.worklet>

namespace dax {
namespace exec {
namespace kernel {

template<class CellType, typename FieldType>
struct SineParameters
{
  dax::exec::WorkMapField<CellType> work;
  dax::exec::Field<FieldType> inField;
  dax::exec::Field<FieldType> outField;
};

template<class Parameters>
struct Sine
{
  DAX_EXEC_EXPORT void operator()(Parameters &parameters,
                                  dax::Id index)
  {
    parameters.work.SetIndex(index);
    dax::worklet::Sine(parameters.work,
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

// Should be templated on grid type too.
template<class GridType, typename FieldType, DAX_DeviceAdapter_TP>
inline void Sine(const GridType &grid,
                 dax::cont::ArrayHandle<FieldType,DeviceAdapter> &inHandle,
                 dax::cont::ArrayHandle<FieldType,DeviceAdapter> &outHandle)
{
  assert(inHandle.GetNumberOfEntries() == outHandle.GetNumberOfEntries());

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
    assert("Number of array entries neither cells nor points.");
    return;
    }

  typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
  GridPackageType gridPackage(grid);

  dax::cont::internal::ExecutionPackageFieldInput<FieldType, DeviceAdapter>
      inField(inHandle, fieldSize);

  dax::cont::internal::ExecutionPackageFieldOutput<FieldType, DeviceAdapter>
      outField(outHandle, fieldSize);

  typedef typename GridPackageType::ExecutionCellType CellType;
  typedef dax::exec::WorkMapField<CellType> WorkType;

  typedef dax::exec::kernel::SineParameters<CellType, FieldType> Parameters;
  Parameters parameters = {
    WorkType(gridPackage.GetExecutionObject()),
    inField.GetExecutionObject(),
    outField.GetExecutionObject()
  };

  DeviceAdapter<void>::Schedule(dax::exec::kernel::Sine<Parameters>(),
                                parameters,
                                fieldSize);
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_Sine_h
