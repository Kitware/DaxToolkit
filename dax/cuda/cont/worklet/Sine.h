/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cuda_cont_worklet_Sine_h
#define __dax_cuda_cont_worklet_Sine_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>
#include <dax/exec/WorkMapField.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cuda/cont/internal/CudaParameters.h>

#include <dax/cuda/exec/ExecutionEnvironment.h>
#include <Worklets/Sine.worklet>

namespace dax {
namespace cuda {
namespace exec {
namespace kernel {

template<typename FieldType>
__global__ void Sine(dax::internal::StructureUniformGrid grid,
                     const dax::internal::DataArray<FieldType> inArray,
                     dax::internal::DataArray<FieldType> outArray)
{
  // TODO: Autoderive this
  typedef dax::exec::WorkMapField<dax::exec::CellVoxel> WorkType;

  WorkType work(grid, 0);
  dax::exec::Field<FieldType> inField(inArray);
  dax::exec::Field<FieldType> outField(outArray);

  // TODO: Consolidate this into function
  dax::Id start = (blockIdx.x * blockDim.x) + threadIdx.x;
  dax::Id increment = gridDim.x;
  dax::Id end = inArray.GetNumberOfEntries();

  for (dax::Id fieldIndex = start; fieldIndex < end; fieldIndex += increment)
    {
    work.SetIndex(fieldIndex);
    dax::worklet::Sine(work, inField, outField);
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
inline void Sine(const dax::cont::UniformGrid &grid,
                 dax::cont::ArrayHandle<FieldType> &inHandle,
                 dax::cont::ArrayHandle<FieldType> &outHandle)
{
  // Determine the cuda parameters from the data structure
  dax::cuda::control::internal::CudaParameters params(grid);

  assert(inHandle.GetNumberOfEntries() == outHandle.GetNumberOfEntries());

  dax::Id numBlocks, numThreads;
  if (inHandle.GetNumberOfEntries() == grid.GetNumberOfPoints())
    {
    numBlocks = params.GetNumberOfPointBlocks();
    numThreads = params.GetNumberOfPointThreads();
    }
  else if (inHandle.GetNumberOfEntries() == grid.GetNumberOfCells())
    {
    numBlocks = params.GetNumberOfCellBlocks();
    numThreads = params.GetNumberOfCellThreads();
    }
  else
    {
    assert("Number of array entries neither cells nor points.");
    return;
    }

  const dax::internal::StructureUniformGrid &structure
      = grid.GetStructureForExecution();
  dax::internal::DataArray<FieldType> inArray = inHandle.ReadyAsInput();
  dax::internal::DataArray<FieldType> outArray = outHandle.ReadyAsOutput();

  dax::cuda::exec::kernel::Sine<<<numBlocks, numThreads>>>(structure,
                                                           inArray,
                                                           outArray);

  outHandle.CompleteAsOutput();
}

}
}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_Sine_h
