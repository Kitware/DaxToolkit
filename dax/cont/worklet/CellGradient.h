/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_worklet_CellGradient_h
#define __dax_cont_worklet_CellGradient_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>
#include <dax/exec/WorkMapCell.h>
#include <dax/exec/internal/FieldBuild.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/Schedule.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <Worklets/CellGradient.worklet>

namespace dax {
namespace exec {
namespace kernel {

template<class CellType>
struct CellGradientParameters
{
  dax::exec::WorkMapCell<CellType> work;
  dax::exec::FieldCoordinates inCoordinates;
  dax::exec::FieldPoint<dax::Scalar> inField;
  dax::exec::FieldCell<dax::Vector3> outField;
};

template<class Parameters>
struct CellGradient {
  DAX_EXEC_EXPORT void operator()(Parameters &parameters,
                                  dax::Id index)
  {
    parameters.work.SetCellIndex(index);
    dax::worklet::CellGradient(parameters.work,
                               parameters.inCoordinates,
                               parameters.inField,
                               parameters.outField);
  }
};

}
}
} // dax::cuda::kernel

namespace dax {
namespace cont {
namespace worklet {

template<class GridType>
inline void CellGradient(const GridType &grid,
                         const typename GridType::Points &points,
                         dax::cont::ArrayHandle<dax::Scalar> &inHandle,
                         dax::cont::ArrayHandle<dax::Vector3> &outHandle)
{
  typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
  GridPackageType gridPackage(grid);

  dax::cont::internal::ExecutionPackageFieldCoordinatesInput<GridType>
      fieldCoordinates(points);

  dax::cont::internal::ExecutionPackageFieldPointInput<dax::Scalar>
      inField(inHandle, grid);

  dax::cont::internal::ExecutionPackageFieldCellOutput<dax::Vector3>
      outField(outHandle, grid);

  typedef typename GridPackageType::ExecutionCellType CellType;
  typedef dax::exec::WorkMapCell<CellType> WorkType;

  typedef dax::exec::kernel::CellGradientParameters<CellType> Parameters;

  Parameters parameters = {
    WorkType(gridPackage.GetExecutionObject()),
    fieldCoordinates.GetExecutionObject(),
    inField.GetExecutionObject(),
    outField.GetExecutionObject()
  };

  dax::cont::schedule(dax::exec::kernel::CellGradient<Parameters>(),
                      parameters,
                      grid.GetNumberOfCells());
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_CellGradient_h
