/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cuda_cont_worklet_Cosine_h
#define __dax_cuda_cont_worklet_Cosine_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>
#include <dax/exec/WorkMapField.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cuda/cont/internal/CudaParameters.h>

#include <dax/cuda/exec/ExecutionEnvironment.h>
#include <Worklets/Cosine.worklet>

namespace dax {
namespace cuda {
namespace exec {
namespace kernel {

template<typename FieldType>
__global__ void Cosine(dax::internal::StructureUniformGrid grid,
                       const dax::internal::DataArray<FieldType> inArray,
                       dax::internal::DataArray<FieldType> outArray)
{
  // TODO: Autoderive this
  typedef dax::exec::WorkMapField<dax::exec::CellVoxel> WorkType;

  WorkType work(grid, 0);
  dax::exec::FieldPoint<FieldType> inField(inArray);
  dax::exec::FieldPoint<FieldType> outField(outArray);

  // TODO: Consolidate this into function
  dax::Id start = (blockIdx.x * blockDim.x) + threadIdx.x;
  dax::Id increment = gridDim.x;
  dax::Id end = dax::internal::numberOfPoints(grid);

  for (dax::Id pointIndex = start; pointIndex < end; pointIndex += increment)
    {
    work.SetIndex(pointIndex);
    dax::worklet::Cosine(work, inField, outField);
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

// Should be templated on grid type too.
template<typename FieldType>
inline void Cosine(const dax::cont::UniformGrid &grid,
                   dax::cont::ArrayHandle<FieldType> &inHandle,
                   dax::cont::ArrayHandle<FieldType> &outHandle)
{
  // Determine the cuda parameters from the data structure
  dax::cuda::control::internal::CudaParameters params(grid.GetNumberOfPoints(),
                                                      grid.GetNumberOfCells());

  dax::cuda::exec::kernel::Cosine
      <<<params.numPointBlocks(), params.numPointThreads()>>>
        (grid.GetStructureForExecution(),
         inHandle.ReadyAsInput(),
         outHandle.ReadyAsOutput());

  outHandle.CompleteAsOutput();
}

}
}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_Cosine_h
