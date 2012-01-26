/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_worklet_testing_FieldMapError_h
#define __dax_cont_worklet_testing_FieldMapError_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>
#include <dax/exec/WorkMapField.h>
#include <dax/exec/internal/ErrorHandler.h>
#include <dax/exec/internal/FieldBuild.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <Worklets/Testing/FieldMapError.worklet>

namespace dax {
namespace exec {
namespace kernel {

template<class CellType>
struct FieldMapErrorParameters
{
  typename CellType::GridStructureType grid;
};

template<class CellType>
struct FieldMapError
{
  DAX_EXEC_EXPORT void operator()(
      FieldMapErrorParameters<CellType> &parameters,
      dax::Id index,
      const dax::exec::internal::ErrorHandler &errorHandler)
  {
    dax::exec::WorkMapField<CellType> work(parameters.grid, errorHandler);
    work.SetIndex(index);
    dax::worklet::testing::FieldMapError(work);
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
inline void FieldMapError(const GridType &grid)
{
  typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
  GridPackageType gridPackage(grid);

  typedef typename GridPackageType::ExecutionCellType CellType;

  typedef dax::exec::kernel::FieldMapErrorParameters<CellType> Parameters;
  Parameters parameters = {
    gridPackage.GetExecutionObject()
  };

  DeviceAdapter::Schedule(
        dax::exec::kernel::FieldMapError<CellType>(),
        parameters,
        grid.GetNumberOfPoints());
}

}
}
}
} //dax::cont::worklet::testing

#endif //__dax_cont_worklet_testing_FieldMapError_h
