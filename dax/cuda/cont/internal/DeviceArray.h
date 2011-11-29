#ifndef DAXDEVICEARRAY_H
#define DAXDEVICEARRAY_H

#include <dax/Types.h>
#include <dax/cont/internal/Macros.h>
#include <dax/internal/DataArray.h>

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <thrust/device_vector.h>

namespace dax {
namespace cont
{
// forward declaration of HostArray
template<typename OtherT> class HostArray;
}
}

namespace dax {
namespace cuda {
namespace cont {
namespace internal {

daxDeclareClassTemplate1(DeviceArray);
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
  DeviceArray(const dax::cuda::cont::internal::DeviceArray<OtherT> &v)
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
  DeviceArray &operator=(const dax::cuda::cont::internal::DeviceArray<OtherT> &v)
  { Parent::operator=(v); return *this; }

  //copy the HostArray on the rhs to this DeviceArray
  template<typename OtherT>
  __host__
  DeviceArray &operator=(const dax::cont::HostArray<OtherT> &v)
  { Parent::operator=(v.Data); return *this;}

  //copy the contents to the passed in host array
  template<typename OtherT>
  __host__
  void toHost(dax::cont::HostArray<OtherT>* v) const
  {
    v->resize(this->size());
    thrust::copy(this->begin(),this->end(),v->begin());
  }

  //set the data array to point to this device vector
  __host__
  ValueType* rawPtr()
  {
    return thrust::raw_pointer_cast(&((*this)[0]));
  }


};

template<typename InputIterator,
         typename OutputIterator>
__host__
OutputIterator toHost(InputIterator first, InputIterator last, OutputIterator dest)
{
  return thrust::copy(first,last,dest);
}

} } } }
#endif // DAXDEVICEARRAY_H
