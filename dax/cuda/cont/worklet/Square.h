/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cuda_cont_worklet_Square_h
#define __dax_cuda_cont_worklet_Square_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>
#include <dax/exec/WorkMapField.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>
#include <dax/cuda/cont/internal/CudaParameters.h>

#include <dax/cuda/exec/ExecutionEnvironment.h>
#include <Worklets/Square.worklet>

namespace dax {
namespace cuda {
namespace exec {
namespace kernel {

template<class CellType, class GridType, typename FieldType>
__global__ void Square(
    const GridType grid,
    const dax::exec::Field<FieldType> inField,
    dax::exec::Field<FieldType> outField)
{
  typedef dax::exec::WorkMapField<CellType> WorkType;

  WorkType work(grid, 0);

  // TODO: Consolidate this into function
  dax::Id start = (blockIdx.x * blockDim.x) + threadIdx.x;
  dax::Id increment = gridDim.x * blockDim.x;
  dax::Id end = inField.GetArray().GetNumberOfEntries();

  for (dax::Id fieldIndex = start; fieldIndex < end; fieldIndex += increment)
    {
    work.SetIndex(fieldIndex);
    dax::worklet::Square(work, inField, outField);
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

template<class GridType, typename FieldType>
inline void Square(const GridType &grid,
                   dax::cont::ArrayHandle<FieldType> &inHandle,
                   dax::cont::ArrayHandle<FieldType> &outHandle)
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

  // Determine the cuda parameters from the data structure
  dax::cuda::control::internal::CudaParameters params(fieldSize);
  dax::Id numBlocks = params.GetNumberOfBlocks();
  dax::Id numThreads = params.GetNumberOfThreads();

  typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
  GridPackageType gridPackage(grid);

  dax::cont::internal::ExecutionPackageFieldInput<FieldType>
      inField(inHandle, fieldSize);

  dax::cont::internal::ExecutionPackageFieldOutput<FieldType>
      outField(outHandle, fieldSize);

  dax::cuda::exec::kernel::Square<typename GridPackageType::ExecutionCellType>
      <<<numBlocks, numThreads>>>(gridPackage.GetExecutionObject(),
                                  inField.GetExecutionObject(),
                                  outField.GetExecutionObject());
}

}
}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_Square_h
