#ifndef EXECUTIVE_H
#define EXECUTIVE_H

#include <cuda.h>
#include <dax/Types.h>
#include <dax/cuda/exec/ExecutionEnvironment.h>
#include <dax/cuda/cont/internal/ManagedDeviceDataArray.h>

#include <vector>

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

template<typename Function>
class Filter
{
public:
  typedef typename Function::OutputDataType OutputDataType;
  typedef dax::cuda::cont::internal::ManagedDeviceDataArrayPtr<OutputDataType> DeviceDataArrayPtr;
  typedef dax::cuda::cont::internal::ManagedDeviceDataArray<OutputDataType> DeviceDataArray;
  DeviceDataArrayPtr OutputData;

  template<typename T>
  Filter(Source<T> src):
    OutputData(new DeviceDataArray() )
  {
    Function::compute(src.Input,OutputData);
  }

  template<typename T, typename U>
  Filter(Source<T> src, Filter<U> conn):
    OutputData(new DeviceDataArray() )
  {
    Function::compute(src.Input,conn.OutputData,OutputData);
  }
};


//we need to pull down the gradient array from the device and store
//it for the user to use on the host, this is just a wrapper
//class to the function that does everything
template<typename T>
void Sink(Filter<T> filter, std::vector<typename T::OutputDataType> &data)
{
  std::cout << "in Sink " << std::endl;
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
  std::cout << "Properly copied vector back to host" << std::endl;
}

}}

#endif // EXECUTIVE_H
