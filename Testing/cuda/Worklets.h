#ifndef __Worklets_h
#define __Worklets_h

#include <cuda.h>
#include <dax/Types.h>

#include <dax/cuda/cont/Modules.h>
#include <dax/cont/StructuredGrid.h>

#include <dax/internal/DataArray.h>
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
template<typename GridType>
__global__ void ComputeElevation(GridType input,
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

template<typename GridType>
//this will need to be auto moc'ed from looking at the worklet
__global__ void ComputeGradient(GridType source,
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
  typedef dax::Scalar OutputType;

  template<typename GridType>
  void run(const dax::cuda::exec::CudaParameters &params,
          GridType& input,
          dax::internal::DataArray<dax::Scalar> output)
    {    
    dax::cont::StructuredGrid *sg =
        dynamic_cast<dax::cont::StructuredGrid*>(&input);
    if(sg)
      {
      dax::internal::StructureUniformGrid grid;
      grid.Origin = sg->Origin;
      grid.Spacing = sg->Spacing;
      grid.Extent.Max = sg->Extent.Max;
      grid.Extent.Min = sg->Extent.Min;

      std::cout << "Execute Elevation Worklet" << std::endl;
      ComputeElevation<<<params.numPointBlocks(),
        params.numPointThreads()>>>(grid,output);
      }
    }
};

struct GradientWorklet
{
  typedef dax::Scalar InputType;
  typedef dax::Vector3 OutputType;

  //this will need to be auto moc'ed from looking at the worklet
  template<typename GridType>
  void run(const dax::cuda::exec::CudaParameters &params,
          GridType& source,
          dax::internal::DataArray<dax::Scalar> input,
          dax::internal::DataArray<dax::Vector3> output)
  {
  dax::cont::StructuredGrid *sg =
      dynamic_cast<dax::cont::StructuredGrid*>(&source);
  if(sg)
    {
    dax::internal::StructureUniformGrid grid;
    grid.Origin = sg->Origin;
    grid.Spacing = sg->Spacing;
    grid.Extent.Max = sg->Extent.Max;
    grid.Extent.Min = sg->Extent.Min;

    std::cout << "Execute Gradient Worklet" << std::endl;
    ComputeGradient<<<params.numPointBlocks(),
      params.numPointThreads()>>>(grid,input,output);
    }
  }
};

namespace modules {

typedef dax::cuda::cont::MapFieldModule<ElevationWorklet> Elevation;
typedef dax::cuda::cont::MapCellModule<GradientWorklet> CellGradient;

}


#endif // WORKLETS_H
