#ifndef DAXDEVICEARRAY_H
#define DAXDEVICEARRAY_H

#include <dax/Types.h>

#include <thrust/device_vector.h>

namespace dax {
namespace cuda {
namespace cont {
namespace internal {

// forward declaration of HostArray
template<typename OtherT> class HostArray;
template<typename T>
class DeviceArray : public thrust::device_vector<T>
{
private:
  typedef thrust::device_vector<T> Parent;

public:
  typedef T  ValueType;

  //empty construtor
  __host__
  DeviceArray(void)
    :Parent() {}

  //create a deviceArray with n elements of value
  __host__
  explicit DeviceArray(std::size_t n, const ValueType &value = ValueType())
    :Parent(n,value) {}

  //copy constructor
  __host__
  DeviceArray(const DeviceArray &v)
    :Parent(v) {}

  //copy constructor with other valueType
  template<typename OtherT>
  __device__
  DeviceArray(const dax::DeviceArray<OtherT> &v)
    :Parent(v) {}

  //copy constructor from a dax::cont::HostArray
  template<typename OtherT>
  __host__
  DeviceArray(const dax::cont::HostArray<OtherT> &v)
    :Parent(v.Data){}

  //build an array from an iterator range
  template<typename InputIterator>
  __host__
  DeviceArray(InputIterator first, InputIterator last)
    :Parent(first,last) {}

  //copy the DeviceArray on the rhs to this DeviceArray
  template<typename OtherT>
  __device__
  DeviceArray &operator=(const DeviceArray<OtherT> &v)
  { Parent::operator=(v); return *this; }

  //copy the HostArray on the rhs to this DeviceArray
  template<typename OtherT>
  __host__
  DeviceArray &operator=(const dax::cont::HostArray<OtherT> &v)
  { Parent::operator=(v.Data); return *this;}

};

typedef DeviceArray<dax::Id> DeviceIdArray;
typedef DeviceArray<dax::Scalar> DeviceScalarArray;
typedef DeviceArray<dax::Vector3> DeviceVector3Array;
typedef DeviceArray<dax::Vector3> DeviceCoordinates;

template<typename T,
         typename OtherT>
__host__
void toHost(dax::DeviceArray<T>& dArray, dax::cont::HostArray<OtherT>& hArray)
{
  hArray.resize(dArray.size());
  thrust::copy(dArray.begin(),dArray.end(),hArray.begin());
}


template<typename InputIterator,
         typename OutputIterator>
__host__
OutputIterator toHost(InputIterator first, InputIterator last, OutputIterator dest)
{
  return thrust::copy(first,last,dest);
}

} } } }
#endif // DAXDEVICEARRAY_H
