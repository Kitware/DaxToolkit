/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef DAXDEVICEARRAY_H
#define DAXDEVICEARRAY_H

#include <dax/Types.h>
#include <dax/cont/internal/Macros.h>
#include <dax/internal/DataArray.h>

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <thrust/device_vector.h>

namespace dax {
namespace cont{
// forward declaration of control classes
template<typename OtherT> class Array;
template<typename OtherT> class ArrayPtr;
template<typename OtherT> class FieldHandleCell;
template<typename OtherT> class FieldHandlePoint;

namespace internal {
template<typename OtherT> class ArrayContainer;
}
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

  //copy constructor from a dax::cont::Array
  template<typename OtherT>
  __host__
  DeviceArray(const dax::cont::Array<OtherT> &v)
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

  //copy the Array on the rhs to this DeviceArray
  template<typename OtherT>
  __host__
  DeviceArray &operator=(const dax::cont::Array<OtherT> &v)
  {Parent::operator=(v.Data); return *this;}

  //copy the contents to the passed in array
  template<typename OtherT>
  __host__
  void toHost(dax::cont::Array<OtherT>* v) const
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

  //this converts an control array an execution array
  //it currently presumes we only have a single
  //type of array in the control side
  DeviceArrayPtr<ValueType> convert(
      boost::shared_ptr<void> controlArray)
    {
    typedef dax::cont::ArrayPtr<ValueType> ArrayPtr;
    typedef dax::cont::Array<ValueType> Array;

    DeviceArrayPtr<ValueType> tempDevice(new DeviceArray<ValueType>());
    ArrayPtr tempCont = boost::static_pointer_cast<Array>(controlArray);
    if(tempCont)
      {
      (*tempDevice) = (*tempCont);
      }
    return tempDevice;
    }
};

template<typename InputIterator,
         typename OutputIterator>
__host__
OutputIterator toHost(InputIterator first, InputIterator last, OutputIterator dest)
{
  return thrust::copy(first,last,dest);
}

template<typename T>
inline DeviceArrayPtr<T> retrieve(const dax::cont::internal::ArrayContainer<T>& container)
  {
  //get device array if it already exists, otherwise
  //convert the control array to device array.
  //need the template keyword to help out some compilers
  //figure out that execution is a templated method
  return container.template arrayExecution<
      dax::cuda::cont::internal::DeviceArray<T> >();
  }

template<typename T>
inline DeviceArrayPtr<T> retrieve(dax::cont::FieldHandleCell<T>& handle)
{
  //get device array if it already exists, otherwise
  //convert the control array to device array.
  return handle.template arrayExecution<
      dax::cuda::cont::internal::DeviceArray<T> >();
}

template<typename T>
inline DeviceArrayPtr<T> retrieve(dax::cont::FieldHandlePoint<T>& handle)
{
  //get device array if it already exists, otherwise
  //convert the control array to device array.

  return handle.template arrayExecution<
      dax::cuda::cont::internal::DeviceArray<T> >();
}

} } } }

#endif // DAXDEVICEARRAY_H
