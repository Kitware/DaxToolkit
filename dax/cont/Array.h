#ifndef __dax_cont_HostArray_h
#define __dax_cont_HostArray_h

#include <vector>
#include <string>

#include <dax/Types.h>
#include <dax/cont/internal/Macros.h>
#include <dax/cont/internal/BaseArray.h>

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

namespace dax {
namespace cuda {
namespace cont {
namespace internal {

// forward declaration of deviceArray
//so we can define it for friend being a friend class
template<typename OtherT> class DeviceArray;

} } } }

namespace dax { namespace cont {

daxDeclareClassTemplate1(Array);
template<typename Type>
class Array :
    public dax::cont::internal::BaseArray

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
    BaseArray(), Name(""), Data()
  {
  }

  explicit Array(const std::size_t n, const ValueType &value = ValueType()):
    BaseArray(), Name(""), Data(n,value)
  {
  }

  Array(const Array &v)
    :Data(v.Data),Name(v.Name) {}

  template<typename OtherT>
  Array(const Array<OtherT> &v)
    :Data(v), Name(v.name()) {}

  template<typename OtherT>
  Array(const std::vector<OtherT> &v)
    :Data(v) {}

  template<typename InputIterator>
  Array(InputIterator first, InputIterator last)
    :Data(first, last) {}

  virtual ~Array(){}

  void setName(const std::string &name) { Name = name; }
  virtual std::string name() const { return Name; }

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
  { Data=(v.Data); Name=v.name();return *this;}

  Array &operator=(const Array &v)
  { Data=(v.Data); Name=v.name(); return *this; }

  //move data from device to host
  template<typename OtherT>
  Array &operator=(const dax::cuda::cont::internal::DeviceArray<OtherT> &v)
  { v.toHost(this); return *this; }

  //move data from device to host
  template<typename OtherT>
  Array &operator=(dax::cuda::cont::internal::DeviceArray<OtherT>* v)
  { v->toHost(this); return *this; }


protected:
  std::string Name;
  Parent Data;
};

} }
#endif // __dax_cont_HostArray_h
