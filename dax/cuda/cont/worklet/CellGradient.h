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
    // TODO: make a pointCoord field thing,
    const dax::internal::DataArray<dax::Scalar> inArray,
    dax::internal::DataArray<dax::Vector3> outArray)
{
  // TODO: Autoderive this
  typedef dax::exec::WorkMapCell<dax::exec::CellVoxel> WorkType;

  WorkType work(grid, 0);
  dax::exec::FieldCoordinates inCoordinates(
        dax::internal::make_DataArrayVector3(NULL, 0));
  dax::exec::FieldPoint<dax::Scalar> inField(inArray);
  dax::exec::FieldCell<dax::Vector3> outField(outArray);

  // TODO: Consolidate this into function
  dax::Id start = (blockIdx.x * blockDim.x) + threadIdx.x;
  dax::Id increment = gridDim.x;
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
                         // TODO: make a pointCoord field thing,
                         dax::cont::ArrayHandle<dax::Scalar> &inHandle,
                         dax::cont::ArrayHandle<dax::Vector3> &outHandle)
{
  // Determine the cuda parameters from the data structure
  dax::cuda::control::internal::CudaParameters params(grid);

  dax::cuda::exec::kernel::CellGradient
      <<<params.GetNumberOfPointBlocks(), params.GetNumberOfPointThreads()>>>
        (grid.GetStructureForExecution(),
         inHandle.ReadyAsInput(),
         outHandle.ReadyAsOutput());

  outHandle.CompleteAsOutput();
}

}
}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_CellGradient_h
