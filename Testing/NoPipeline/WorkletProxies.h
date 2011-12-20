#ifndef WORKLETPROXIES_H
#define WORKLETPROXIES_H

#include <dax/internal/ExportMacros.h>
#include <dax/Types.h>

#include <cuda.h>
#include <dax/Types.h>

#include <dax/cont/StructuredGrid.h>

#include <dax/internal/DataArray.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapCell.h>
#include <dax/exec/WorkMapField.h>
#include <dax/exec/internal/FieldBuild.h>
#include <dax/cuda/exec/ExecutionEnvironment.h>
#include <dax/cuda/exec/CudaParameters.h>

#include <dax/cuda/cont/internal/DeviceArray.h>

#include <Worklets/CellGradient.worklet>
#include <Worklets/Cosine.worklet>
#include <Worklets/Elevation.worklet>
#include <Worklets/Sine.worklet>
#include <Worklets/Square.worklet>


namespace workletFunctions
{
template<typename WorkType>
__device__ void computeIteration(dax::Id size, dax::Id &start,
                                 dax::Id &end, dax::Id &inc)
{
  start = (blockIdx.x * blockDim.x) + threadIdx.x;
  inc = gridDim.x;
  end = size;

}

template<typename T, typename U>
__global__ void ElevationFunction(dax::Id size, dax::internal::StructureUniformGrid g,
                                  dax::internal::DataArray<T> in,
                                  dax::internal::DataArray<U> out)
{
  //this is need to be auto derived
  typedef dax::exec::WorkMapField<dax::exec::CellVoxel> WorkType;

  dax::Id start,end,inc;
  computeIteration<WorkType>(size,start,end,inc);

  WorkType work(g,start);

  //convert the device arrays to data arrays and than into fields
  //this should be cleaned up somehow
  const dax::exec::FieldCoordinates inField(in);
  dax::exec::FieldPoint<U> outField(out);

  for(;start < end; start += inc)
    {
    work.SetIndex(start);
    Elevation(work, inField, outField);
    }
}


template<typename T, typename U>
__global__ void SquareFunction(dax::Id size, dax::internal::StructureUniformGrid g,
                                  dax::internal::DataArray<T> in,
                                  dax::internal::DataArray<U> out)
{
  //this is need to be auto derived
  typedef dax::exec::WorkMapField<dax::exec::CellVoxel> WorkType;

  dax::Id start,end,inc;
  computeIteration<WorkType>(size,start,end,inc);

  WorkType work(g,start);

  //convert the device arrays to data arrays and than into fields
  //this should be cleaned up somehow
  const dax::exec::FieldPoint<T> inField(in);
  dax::exec::FieldPoint<U> outField(out);

  for(;start < end; start += inc)
    {
    work.SetIndex(start);
    Square(work, inField, outField);
    }
}

}
namespace workletProxies
{

dax::internal::StructureUniformGrid convertDataSet(dax::cont::StructuredGrid &sg)
  {
  dax::internal::StructureUniformGrid grid;
  grid.Origin = sg.Origin;
  grid.Spacing = sg.Spacing;
  grid.Extent.Max = sg.Extent.Max;
  grid.Extent.Min = sg.Extent.Min;

  return grid;
  }

class Elevation
{
public:
  template<typename G, typename T, typename U>
  void operator()(G &g,
                  dax::cont::HostArray<T>& in,
                  U& out)
  {
    typedef typename U::DataType OutType;

    //convert from host to device arrayss
    dax::cuda::cont::internal::DeviceArray<T> ind(in);
    dax::cuda::cont::internal::DeviceArray<OutType> outd(out.array());

    //determine the cuda parameters from the data structure
    dax::cuda::exec::CudaParameters params(g);

    //this type needs to be auto derived
    dax::internal::StructureUniformGrid grid = convertDataSet(g);

    //this should be easier to do
    dax::internal::DataArray<T> inField(ind);
    dax::internal::DataArray<OutType> outField(outd);

    //in and out are automatically converted to the correct type
    //by explicit constructors on FieldPoint and FieldCoordinates
    workletFunctions::ElevationFunction<<<params.numPointBlocks(),
        params.numPointThreads()>>>(static_cast<dax::Id>(g.numPoints()),grid,
                                    inField,outField);

   //move the results back from the device to the host
    out.array() = outd;

    //associate the out variable array with the datastructure g
    out.associate(g);
  }
};

class Square
{
public:
  template<typename G, typename T, typename U>
  void operator()(G &g,
                  dax::cont::HostArray<T>& in,
                  U& out)
  {
    typedef typename U::DataType OutType;

    //convert from host to device arrayss
    dax::cuda::cont::internal::DeviceArray<T> ind(in);
    dax::cuda::cont::internal::DeviceArray<OutType> outd(out.array());

    //determine the cuda parameters from the data structure
    dax::cuda::exec::CudaParameters params(g);

    //this type needs to be auto derived
    dax::internal::StructureUniformGrid grid = convertDataSet(g);

    //this should be easier to do
    dax::internal::DataArray<T> inField(ind);
    dax::internal::DataArray<OutType> outField(outd);

    //in and out are automatically converted to the correct type
    //by explicit constructors on FieldPoint and FieldCoordinates
    workletFunctions::SquareFunction<<<params.numPointBlocks(),
        params.numPointThreads()>>>(static_cast<dax::Id>(g.numPoints()),grid,
                                    inField,outField);

   //move the results back from the device to the host
    out.array() = outd;

    //associate the out variable array with the datastructure g
    out.associate(g);
  }
};

}



#endif // WORKLETPROXIES_H

