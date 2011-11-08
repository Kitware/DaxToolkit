#ifndef __Worklets_h
#define __Worklets_h

#include <cuda.h>
#include <dax/Types.h>

#include <dax/cuda/cont/Modules.h>

#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapCell.h>
#include <dax/exec/WorkMapField.h>
#include <dax/exec/internal/FieldBuild.h>
#include <dax/cuda/cont/Modules.h>
#include <dax/cuda/exec/ExecutionEnvironment.h>
#include <dax/cuda/exec/CudaParameters.h>

#include <Worklets/CellGradient.worklet>
#include <Worklets/Cosine.worklet>
#include <Worklets/Elevation.worklet>
#include <Worklets/Sine.worklet>
#include <Worklets/Square.worklet>

//this will need to be auto moc'ed from looking at the worklet
__global__ void ComputeElevation(dax::internal::StructureUniformGrid input,
                               dax::internal::DataArray<dax::Scalar> output)
{
  dax::Id pointIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
  dax::Id pointIncrement = gridDim.x;
  dax::Id numPoints = dax::internal::numberOfPoints(input);

  dax::exec::WorkMapField<dax::exec::CellVoxel> work(input, pointIndex);
  dax::exec::FieldCoordinates pointCoord
      = dax::exec::internal::fieldCoordinatesBuild(input);
  dax::exec::FieldPoint<dax::Scalar> outField(output);

  for ( ; pointIndex < numPoints; pointIndex += pointIncrement)
    {
    work.SetIndex(pointIndex);
    Elevation(work, pointCoord, outField);
    }
}

//this will need to be auto moc'ed from looking at the worklet
__global__ void ComputeGradient(dax::internal::StructureUniformGrid source,
                              dax::internal::DataArray<dax::Scalar> input,
                              dax::internal::DataArray<dax::Vector3> output)
{
  dax::Id cellIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
  dax::Id cellIncrement = gridDim.x;
  dax::Id numCells = dax::internal::numberOfCells(source);

  dax::exec::WorkMapCell<dax::exec::CellVoxel> work(source, cellIndex);
  dax::exec::FieldCoordinates pointCoord
      = dax::exec::internal::fieldCoordinatesBuild(source);
  dax::exec::FieldPoint<dax::Scalar> inPointField(input);
  dax::exec::FieldCell<dax::Vector3> outCellField(output);

  for ( ; cellIndex < numCells; cellIndex += cellIncrement)
    {
    work.SetCellIndex(cellIndex);
    CellGradient(work, pointCoord, inPointField, outCellField);
    }
}

struct ElevationWorklet
{
  typedef dax::internal::StructureUniformGrid ModelType;
  typedef dax::Scalar OutputType;

  void run(const dax::cuda::exec::CudaParameters &params,
          dax::internal::StructureUniformGrid input,
          dax::internal::DataArray<dax::Scalar> output)
    {
    std::cout << "Execute Elevation Worklet" << std::endl;
    ComputeElevation<<<params.numPointBlocks(),
        params.numPointThreads()>>>(input,output);
    }
};

struct GradientWorklet
{  
  typedef dax::internal::StructureUniformGrid ModelType;
  typedef dax::Scalar InputType;
  typedef dax::Vector3 OutputType;

  //this will need to be auto moc'ed from looking at the worklet
  void run(const dax::cuda::exec::CudaParameters &params,
          dax::internal::StructureUniformGrid source,
          dax::internal::DataArray<dax::Scalar> input,
          dax::internal::DataArray<dax::Vector3> output)
  {
  std::cout << "Execute Gradient Worklet" << std::endl;
  ComputeGradient<<<params.numPointBlocks(),
      params.numPointThreads()>>>(source,input,output);
  }
};

namespace modules {

typedef dax::cuda::cont::MapFieldModule<ElevationWorklet> Elevation;
typedef dax::cuda::cont::MapCellModule<GradientWorklet> CellGradient;

}


#endif // WORKLETS_H
