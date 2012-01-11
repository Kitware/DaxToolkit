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
#include <dax/cont/UniformGrid.h>
#include <dax/cuda/cont/internal/CudaParameters.h>

#include <dax/cuda/exec/ExecutionEnvironment.h>
#include <Worklets/CellGradient.worklet>

namespace dax {
namespace cuda {
namespace exec {
namespace kernel {

__global__ void CellGradient(
    dax::internal::StructureUniformGrid grid,
    const dax::exec::FieldCoordinates inCoordinates,
    const dax::exec::FieldPoint<dax::Scalar> inField,
    dax::exec::FieldCell<dax::Vector3> outField)
{
  // TODO: Autoderive this
  typedef dax::exec::WorkMapCell<dax::exec::CellVoxel> WorkType;

  WorkType work(grid, 0);

  // TODO: Consolidate this into function
  dax::Id start = (blockIdx.x * blockDim.x) + threadIdx.x;
  dax::Id increment = gridDim.x * blockDim.x;
  dax::Id end = dax::internal::numberOfCells(grid);

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

// Should be templated on grid type.
inline void CellGradient(const dax::cont::UniformGrid &grid,
                         const dax::cont::UniformGrid::Points &points,
                         dax::cont::ArrayHandle<dax::Scalar> &inHandle,
                         dax::cont::ArrayHandle<dax::Vector3> &outHandle)
{
  // Determine the cuda parameters from the data structure
  dax::cuda::control::internal::CudaParameters params(grid);

  dax::Id numBlocks = params.GetNumberOfPointBlocks();
  dax::Id numThreads = params.GetNumberOfPointThreads();

  const dax::internal::StructureUniformGrid &structure
      = grid.GetStructureForExecution();

  dax::exec::FieldCoordinates fieldCoordinates
      = dax::exec::internal::fieldCoordinatesBuild(points.GetStructureForExecution());

  dax::internal::DataArray<dax::Scalar> inArray = inHandle.ReadyAsInput();
  dax::exec::FieldPoint<dax::Scalar> inField(inArray);

  dax::internal::DataArray<dax::Vector3> outArray = outHandle.ReadyAsOutput();
  dax::exec::FieldCell<dax::Vector3> outField(outArray);

  dax::cuda::exec::kernel::CellGradient<<<numBlocks, numThreads>>>(structure,
                                                                   fieldCoordinates,
                                                                   inField,
                                                                   outField);

  outHandle.CompleteAsOutput();
}

}
}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_CellGradient_h
