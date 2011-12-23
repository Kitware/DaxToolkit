#ifndef __dax_cont_HostArray_h
#define __dax_cont_HostArray_h


#include <vector>
#include <string>

#include <dax/Types.h>
#include <dax/cont/internal/Macros.h>
#include <dax/cont/internal/ArrayContainer.h>

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

namespace dax {
namespace cuda {
namespace cont {
namespace internal {

// forward declaration of deviceArray
//so we can define it for friend being a friend class
template<typename OtherT> class DeviceArray;
template<typename OtherT> class DeviceArrayPtr;
} } } }

namespace dax { namespace cont {

daxDeclareClassTemplate1(Array);
template<typename Type>
class Array
{
private:
  typedef std::vector<Type> Parent;

public:
  typedef Type ValueType;

  typedef typename Parent::iterator iterator;
  typedef typename Parent::const_iterator const_iterator;

  //the device array needs to be a friend class
  //so that it can copy the real data in the host array
  template <class OtherT> friend class dax::cuda::cont::internal::DeviceArray;

  Array(void):
  Data()
  {
  }

  explicit Array(const std::size_t n, const ValueType &value = ValueType()):
    Data(n,value)
  {
  }

  Array(const Array &v)
    :Data(v.Data)
    {}

  template<typename OtherT>
  Array(const Array<OtherT> &v)
    :Data(v)
    {}

  template<typename OtherT>
  Array(const std::vector<OtherT> &v)
    :Data(v) {}

  template<typename InputIterator>
  Array(InputIterator first, InputIterator last)
    :Data(first, last) {}

  virtual ~Array(){}

  std::size_t size() const { return this->Data.size(); }
  std::size_t capacity() const { return this->Data.capacity(); }

  void reserve(const std::size_t& sz) { this->Data.reserve(sz); }
  void resize(const std::size_t& sz, ValueType t=ValueType() ) { this->Data.resize(sz,t); }

  const_iterator begin() const { return this->Data.begin(); }
  const_iterator end() const { return this->Data.end(); }

  iterator begin() { return this->Data.begin(); }
  iterator end() { return this->Data.end(); }

  void push_back(const ValueType& v) { this->Data.push_back(v); }

  ValueType operator [](const std::size_t& idx) const { return this->Data[idx]; }
  ValueType& operator [](const std::size_t& idx) { return this->Data[idx]; }
  ValueType at(const std::size_t& idx) const { return this->Data.at(idx); }

  template<typename OtherT>
  Array &operator=(const std::vector<OtherT> &v)
  { Data=(v.Data); return *this;}

  template<typename OtherT>
  Array &operator=(const Array<OtherT> &v)
  { Data=(v.Data);return *this;}

  Array &operator=(const Array &v)
  { Data=(v.Data); return *this; }

  //move data from device to host
  template<typename OtherT>
  Array &operator=(const dax::cuda::cont::internal::DeviceArray<OtherT> &v)
  { v.toHost(this); return *this; }

  //move data from device to host
  template<typename OtherT>
  Array &operator=(dax::cuda::cont::internal::DeviceArray<OtherT>* v)
  { v->toHost(this); return *this; }

  //this converts a execution array to a control array
  //it currently presumes we only have a single
  //type of array in the execution side
  ArrayPtr<ValueType> convert(
      boost::shared_ptr<void> execArray)
    {
    typedef dax::cuda::cont::internal::DeviceArrayPtr<ValueType> DevArrayPtr;
    typedef dax::cuda::cont::internal::DeviceArray<ValueType> DevArray;

    ArrayPtr<ValueType> tempControl(new Array<ValueType>());
    DevArrayPtr tempDevice = boost::static_pointer_cast<DevArray>(execArray);
    (*tempControl) = (*tempDevice);
    return tempControl;
    }

protected:
  Parent Data;
};

template<typename T>
dax::cont::ArrayPtr<T> retrieve(dax::cont::internal::ArrayContainer<T>& container)
{
  return container.template arrayControl<dax::cont::Array<T> >();
}

} }
#endif // __dax_cont_HostArray_h
