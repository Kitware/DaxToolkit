/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cuda_cont_worklet_Elevation_h
#define __dax_cuda_cont_worklet_Elevation_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>
#include <dax/exec/WorkMapField.h>
#include <dax/exec/internal/FieldBuild.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cuda/cont/internal/CudaParameters.h>

#include <dax/cuda/exec/ExecutionEnvironment.h>
#include <Worklets/Elevation.worklet>

namespace dax {
namespace cuda {
namespace exec {
namespace kernel {

__global__ void Elevation(dax::internal::StructureUniformGrid grid,
                          const dax::exec::FieldCoordinates inCoordinates,
                          dax::exec::FieldPoint<dax::Scalar> outField)
{
  // TODO: Autoderive this
  typedef dax::exec::WorkMapField<dax::exec::CellVoxel> WorkType;

  WorkType work(grid, 0);

  // TODO: Consolidate this into function
  dax::Id start = (blockIdx.x * blockDim.x) + threadIdx.x;
  dax::Id increment = gridDim.x * blockDim.x;
  dax::Id end = dax::internal::numberOfPoints(grid);

  for (dax::Id pointIndex = start; pointIndex < end; pointIndex += increment)
    {
    work.SetIndex(pointIndex);
    dax::worklet::Elevation(work, inCoordinates, outField);
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
inline void Elevation(const dax::cont::UniformGrid &grid,
                      const dax::cont::UniformGrid::Points &points,
                      dax::cont::ArrayHandle<dax::Scalar> &outHandle)
{
  // Determine the cuda parameters from the data structure
  dax::cuda::control::internal::CudaParameters params(grid);

  dax::Id numBlocks = params.GetNumberOfPointBlocks();
  dax::Id numThreads = params.GetNumberOfPointThreads();

  const dax::internal::StructureUniformGrid &structure
      = grid.GetStructureForExecution();

  dax::exec::FieldCoordinates fieldCoordinates
      = dax::exec::internal::fieldCoordinatesBuild(points.GetStructureForExecution());

  dax::internal::DataArray<dax::Scalar> outArray = outHandle.ReadyAsOutput();
  dax::exec::FieldPoint<dax::Scalar> outField(outArray);

  dax::cuda::exec::kernel::Elevation<<<numBlocks, numThreads>>>(structure,
                                                                fieldCoordinates,
                                                                outField);

  outHandle.CompleteAsOutput();
}

}
}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_Elevation_h
