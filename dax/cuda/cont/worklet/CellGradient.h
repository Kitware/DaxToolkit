/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cuda_cont_worklet_CellGradient_h
#define __dax_cuda_cont_worklet_CellGradient_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>
#include <dax/exec/WorkMapCell.h>
#include <dax/exec/internal/FieldBuild.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>
#include <dax/cuda/cont/internal/CudaParameters.h>

#include <dax/cuda/exec/ExecutionEnvironment.h>
#include <Worklets/CellGradient.worklet>

namespace dax {
namespace cuda {
namespace exec {
namespace kernel {

template<class CellType, class GridType>
__global__ void CellGradient(const GridType grid,
                             const dax::exec::FieldCoordinates inCoordinates,
                             const dax::exec::FieldPoint<dax::Scalar> inField,
                             dax::exec::FieldCell<dax::Vector3> outField)
{
  typedef dax::exec::WorkMapCell<CellType> WorkType;

  WorkType work(grid, 0);

  // TODO: Consolidate this into function
  dax::Id start = (blockIdx.x * blockDim.x) + threadIdx.x;
  dax::Id increment = gridDim.x * blockDim.x;
  dax::Id end = outField.GetArray().GetNumberOfEntries();

  for (dax::Id cellIndex = start; cellIndex < end; cellIndex += increment)
    {
    work.SetCellIndex(cellIndex);
    dax::worklet::CellGradient(work, inCoordinates, inField, outField);
    }
}

}
}
}
} // dax::cuda::exec::kernel

namespace dax {
namespace cuda {
namespace cont {
namespace worklet {

template<class GridType>
inline void CellGradient(const GridType &grid,
                         const typename GridType::Points &points,
                         dax::cont::ArrayHandle<dax::Scalar> &inHandle,
                         dax::cont::ArrayHandle<dax::Vector3> &outHandle)
{
  // Determine the cuda parameters from the data structure
  dax::cuda::control::internal::CudaParameters params(grid);

  dax::Id numBlocks = params.GetNumberOfPointBlocks();
  dax::Id numThreads = params.GetNumberOfPointThreads();

  typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
  GridPackageType gridPackage(grid);

  dax::cont::internal::ExecutionPackageFieldCoordinatesInput<GridType>
      fieldCoordinates(points);

  dax::cont::internal::ExecutionPackageFieldPointInput<dax::Scalar>
      inField(inHandle, grid);

  dax::cont::internal::ExecutionPackageFieldCellOutput<dax::Vector3>
      outField(outHandle, grid);

  dax::cuda::exec::kernel::CellGradient<typename GridPackageType::ExecutionCellType>
      <<<numBlocks, numThreads>>>(gridPackage.GetExecutionObject(),
                                  fieldCoordinates.GetExecutionObject(),
                                  inField.GetExecutionObject(),
                                  outField.GetExecutionObject());
}

}
}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_CellGradient_h
