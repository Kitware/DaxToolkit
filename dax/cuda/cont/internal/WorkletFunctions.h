#ifndef __dax_cuda_exec_WorkletFunctions_h
#define __dax_cuda_exec_WorkletFunctions_h

#include <cuda.h>
#include <dax/Types.h>
#include <dax/internal/ExportMacros.h>

#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapCell.h>
#include <dax/exec/WorkMapField.h>

#include <dax/cuda/exec/ExecutionEnvironment.h> //needed for DAX_WORKLET
#include <Worklets/CellGradient.worklet>
#include <Worklets/Cosine.worklet>
#include <Worklets/Elevation.worklet>
#include <Worklets/Sine.worklet>
#include <Worklets/Square.worklet>

namespace dax {
namespace cuda {
namespace cont {
namespace internal {



template<typename WorkType>
DAX_EXEC_CONT_EXPORT void computeIteration(dax::Id size, dax::Id &start,
                                 dax::Id &end, dax::Id &inc)
{
  start = (blockIdx.x * blockDim.x) + threadIdx.x;
  inc = gridDim.x;
  end = size;

}


template<typename T, typename T2, typename U>
__global__ void CellGradientFunction(dax::Id size,
                                     dax::internal::StructureUniformGrid g,
                                    dax::internal::DataArray<T> in,
                                    dax::internal::DataArray<T2> in2,
                                    dax::internal::DataArray<U> out)
{
  //this is need to be auto derived
  typedef dax::exec::WorkMapCell<dax::exec::CellVoxel> WorkType;

  dax::Id start,end,inc;
  computeIteration<WorkType>(size,start,end,inc);

  WorkType work(g,start);

  //convert the device arrays to data arrays and than into fields
  //this should be cleaned up somehow
  const dax::exec::FieldCoordinates pField(in);
  const dax::exec::FieldPoint<T2> inField(in2);

  dax::exec::FieldCell<U> outField(out);

  for(;start < end; start += inc)
    {
    //this is really bad api, should be same as work in my mind
    work.SetCellIndex(start);
    CellGradient(work, pField, inField, outField);
    }
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
  const dax::exec::Field<T> inField(in);
  dax::exec::Field<U> outField(out);

  for(;start < end; start += inc)
    {
    work.SetIndex(start);
    Square(work, inField, outField);
    }
}


template<typename T, typename U>
__global__ void SineFunction(dax::Id size, dax::internal::StructureUniformGrid g,
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
  const dax::exec::Field<T> inField(in);
  dax::exec::Field<U> outField(out);

  for(;start < end; start += inc)
    {
    work.SetIndex(start);
    Sine(work, inField, outField);
    }
}


template<typename T, typename U>
__global__ void CosineFunction(dax::Id size, dax::internal::StructureUniformGrid g,
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
  const dax::exec::Field<T> inField(in);
  dax::exec::Field<U> outField(out);

  for(;start < end; start += inc)
    {
    work.SetIndex(start);
    Cosine(work, inField, outField);
    }
}

} } } }
#endif // __dax_cuda_exec_WorkletFunctions_h
