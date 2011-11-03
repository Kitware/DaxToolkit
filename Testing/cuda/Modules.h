#ifndef MODULES_H
#define MODULES_H

#include <cuda.h>

// Includes for host code.
#include <dax/cuda/cont/internal/ManagedDeviceDataArray.h>

// Includes for device code.
#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapCell.h>
#include <dax/exec/WorkMapField.h>
#include <dax/exec/internal/FieldBuild.h>
#include <dax/cuda/exec/ExecutionEnvironment.h>

//Worklets
#include <Worklets/Elevation.worklet>
#include <Worklets/CellGradient.worklet>

#include <vector>


namespace cudaExecution
{
  class CudaParameters
  {
    //holds the basic information needed to determine the
    //number of threads and blocks to run on the current cuda card
  public:
    template< typename T>
    CudaParameters(T source):
      NumPointThreads(128),
      NumCellThreads(128)
    {
      dax::Id numPts = dax::internal::numberOfPoints(source);
      dax::Id numCells = dax::internal::numberOfCells(source);

      NumPointBlocks = (numPts+NumPointThreads-1)/NumPointThreads;
      NumCellBlocks = (numCells+NumPointThreads-1)/NumPointThreads;
    }

    dax::Id numPointBlocks() const { return NumPointBlocks; }
    dax::Id numPointThreads() const { return NumPointThreads; }

    dax::Id numCellBlocks() const { return NumCellBlocks; }
    dax::Id numCellThreads() const { return NumCellThreads; }
  protected:
    dax::Id NumPointBlocks;
    dax::Id NumPointThreads;
    dax::Id NumCellBlocks;
    dax::Id NumCellThreads;
  };

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
  void run(const cudaExecution::CudaParameters &params,
          dax::internal::StructureUniformGrid input,
          dax::internal::DataArray<dax::Scalar> output)
       {
         ComputeElevation<<<params.numPointBlocks(),
                 params.numPointThreads()>>>(input,output);
       }
};

struct GradientWorklet
{
  //this will need to be auto moc'ed from looking at the worklet
  void run(const cudaExecution::CudaParameters &params,
          dax::internal::StructureUniformGrid source,
          dax::internal::DataArray<dax::Scalar> input,
          dax::internal::DataArray<dax::Vector3> output)
  {
  ComputeGradient<<<params.numPointBlocks(),
          params.numPointThreads()>>>(source,input,output);
  }
};

}
namespace dax { namespace modules {

template< class Derived >
class ModuleBase
{
public:
  ModuleBase():AlreadyComputed(false){}
  void executeComputation()
    {
    if(!AlreadyComputed)
      {
      static_cast<Derived*>(this)->executeComputation();
      AlreadyComputed=true;
      }
    else
      {
      }
    }
protected:
  bool AlreadyComputed;

};

template< typename SourceType,
          typename OutputType,
          typename Worklet>
class MapFieldModule : public ModuleBase< MapFieldModule<SourceType,OutputType,Worklet> >
{
public:
  typedef OutputType OutputDataType;
  typedef SourceType SourceDataType;
  typedef dax::cuda::cont::internal::ManagedDeviceDataArrayPtr
                                <OutputDataType> DeviceDataArrayPtr;

  SourceDataType *Source;
  DeviceDataArrayPtr Output;

  MapFieldModule(SourceDataType& source, DeviceDataArrayPtr output):
    Source(&source),Output(output)
    {}

  void executeComputation()
    {
    const cudaExecution::CudaParameters params(*this->Source);
    this->Output->Allocate(dax::internal::numberOfPoints(*this->Source));
    Worklet().run(params,*this->Source,this->Output->GetArray());
    }
};

template< typename SourceType,
          typename InputType,
          typename OutputType,
          typename Worklet>
class MapCellModule : public ModuleBase< MapCellModule<SourceType,InputType,OutputType,Worklet> >
{
public:
  typedef SourceType SourceDataType;
  typedef InputType  InputDataType;
  typedef OutputType OutputDataType;

  typedef dax::cuda::cont::internal::ManagedDeviceDataArrayPtr
                                      <InputDataType> InputDeviceDataArrayPtr;
  typedef dax::cuda::cont::internal::ManagedDeviceDataArrayPtr
                                      <OutputDataType> OutputDeviceDataArrayPtr;

  SourceDataType *Source;
  InputDeviceDataArrayPtr Input;
  OutputDeviceDataArrayPtr Output;

  MapCellModule(SourceType& source, InputDeviceDataArrayPtr input,
                OutputDeviceDataArrayPtr output):
    Source(&source),Input(input),Output(output)
    {}

  void executeComputation()
    {
    const cudaExecution::CudaParameters params(*this->Source);
    this->Output->Allocate(dax::internal::numberOfCells(*this->Source));
    Worklet().run(params,*this->Source,
                         this->Input->GetArray(),
                         this->Output->GetArray());
    }
};

//ElevationM and GraidentM can be typedefs,
//which mean in theory all user defined classes will just be typedefs
//ToDo: push the function pointer into the template definition, rather than hard coded
typedef MapFieldModule <dax::internal::StructureUniformGrid,dax::Scalar, cudaExecution::ElevationWorklet> ElevationM;
typedef MapCellModule <dax::internal::StructureUniformGrid,dax::Scalar,dax::Vector3,cudaExecution::GradientWorklet> GradientM;

}}

#endif // MODULES_H
