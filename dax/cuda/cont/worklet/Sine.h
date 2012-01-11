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
#include <dax/cont/internal/ExecutionPackageGrid.h>
#include <dax/cuda/cont/internal/CudaParameters.h>

#include <dax/cuda/exec/ExecutionEnvironment.h>
#include <Worklets/Sine.worklet>

namespace dax {
namespace cuda {
namespace exec {
namespace kernel {

template<class GridType, typename FieldType>
__global__ void Sine(
    dax::cont::internal::ExecutionPackageGrid<GridType> grid,
    const dax::exec::Field<FieldType> inField,
    dax::exec::Field<FieldType> outField)
{
  typedef dax::cont::internal::ExecutionPackageGrid<GridType> PackageGrid;
  typedef typename PackageGrid::ExecutionCellType CellType;
  typedef dax::exec::WorkMapField<CellType> WorkType;

  WorkType work(grid, 0);

  // TODO: Consolidate this into function
  dax::Id start = (blockIdx.x * blockDim.x) + threadIdx.x;
  dax::Id increment = gridDim.x * blockDim.x;
  dax::Id end = inField.GetArray().GetNumberOfEntries();

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
template<class GridType, typename FieldType>
inline void Sine(const GridType &grid,
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

  dax::cont::internal::ExecutionPackageGrid<GridType> gridPackage(grid);

  dax::internal::DataArray<FieldType> inArray = inHandle.ReadyAsInput();
  dax::exec::Field<FieldType> inField(inArray);

  dax::internal::DataArray<FieldType> outArray = outHandle.ReadyAsOutput();
  dax::exec::Field<FieldType> outField(outArray);

  dax::cuda::exec::kernel::Sine<<<numBlocks, numThreads>>>(gridPackage,
                                                           inField,
                                                           outField);

  outHandle.CompleteAsOutput();
}

}
}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_Sine_h
