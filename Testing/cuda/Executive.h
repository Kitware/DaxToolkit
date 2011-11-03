#ifndef EXECUTIVE_H
#define EXECUTIVE_H

#include <cuda.h>
#include <dax/Types.h>
#include <dax/cuda/exec/ExecutionEnvironment.h>
#include <dax/cuda/cont/internal/ManagedDeviceDataArray.h>

#include <vector>

#include "boost/function.hpp"
#include "boost/bind.hpp"
#include <typeinfo>  //for 'typeid' to work

namespace{
void no_dependency()
  {
  //we generate an empty function that we can bind too
  //when we have no dependent filters.
  };
}

namespace dax { namespace exec {

template< typename T>
class Source
{
public:
  typedef T Type;
  Source(T input):
    Input(input)
    {
    }
  T Input;
};

template<typename Module>
class Filter
{
public:
  typedef typename Module::OutputDataType OutputDataType;
  typedef dax::cuda::cont::internal::ManagedDeviceDataArrayPtr<OutputDataType> DeviceDataArrayPtr;
  typedef dax::cuda::cont::internal::ManagedDeviceDataArray<OutputDataType> DeviceDataArray;

  DeviceDataArrayPtr OutputData;
  Module M;
  boost::function<void(void)> Dependency;

  template<typename T>
  Filter(Source<T> &src):
    OutputData(new DeviceDataArray() ),
    M(src.Input,OutputData),
    Dependency(boost::bind(&no_dependency))
  {
  }

  template<typename T, typename U>
  Filter(Source<T> &src, Filter<U>  &conn):
    OutputData(new DeviceDataArray() ),
    M(src.Input,conn.OutputData,OutputData),
    Dependency(boost::bind(&Filter<U>::compute,&conn))
  {
  }

  void compute()
    {
    //this is does the actual computation
    //Dependency call makes sure we do a depth first walk
    //up the pipeline
    Dependency();
    M.executeComputation();
    }
};

//we need to pull down the gradient array from the device and store
//it for the user to use on the host
template<typename T>
void  Sink(Filter<T> filter, std::vector<typename T::OutputDataType> &data)
{
  std::cout << "In Sink" << std::endl;
  filter.compute();

  if (cudaThreadSynchronize() != cudaSuccess)
    {
    std::cout << "failed to synchronize the cuda device" << std::endl;
    abort();
    }

  //make the data the correct size
  data.resize(filter.OutputData->size());

  typedef typename T::OutputDataType dataType;
  dax::internal::DataArray<dataType> dataArray;
  dataArray.SetPointer(&data[0],data.size());

  filter.OutputData->CopyToHost(dataArray);
  std::cout << "Properly copied data back to host" << std::endl;
}


}}

#endif // EXECUTIVE_H
