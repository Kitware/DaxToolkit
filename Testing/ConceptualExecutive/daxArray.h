#ifndef DAXARRAY_H
#define DAXARRAY_H

#include <vector>
#include <string>

#include "daxTypes.h"
#include "boost/iterator/counting_iterator.hpp"
#include "boost/iterator/transform_iterator.hpp"

//forward declerations
namespace dax {

// forward declaration of HostArray
template<typename Type> class HostArray;

// forward declaration of deviceArray
template<typename OtherT> class DeviceArray;

}

namespace dax { namespace internal {

class BaseArray
{
public:
  virtual ~BaseArray(){}
  virtual std::string name() const=0;
protected:
    std::string Name;
  };
} }

namespace dax {

template<typename Type>
class HostArray :
    public dax::internal::BaseArray

{
private:
  typedef std::vector<Type> Parent;

public:
  typedef Type ValueType;
  typedef typename Parent::iterator iterator;
  typedef typename Parent::const_iterator const_iterator;

  //the device array needs to be a friend class
  //so that it can copy the real data in the host array
  template <class OtherT> friend class DeviceArray;

  HostArray(void):
    BaseArray(), Name(""), Data()
  {
  }

  explicit HostArray(const std::size_t n, const ValueType &value = ValueType()):
    BaseArray(), Name(""), Data(n,value)
  {
  }

  HostArray(const HostArray &v)
    :Data(v.Data),Name(v.Name) {}

  template<typename OtherT>
  HostArray(const HostArray<OtherT> &v)
    :Data(v), Name(v.name()) {}

  template<typename OtherT>
  HostArray(const std::vector<OtherT> &v)
    :Data(v) {}

  template<typename InputIterator>
  HostArray(InputIterator first, InputIterator last)
    :Data(first, last) {}

  virtual ~HostArray(){}

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

  template<typename T>
  ValueType operator [](const T& idx) const { return this->Data[idx]; }

  template<typename T>
  ValueType& operator [](const T& idx) { return this->Data[idx]; }

  template<typename T>
  ValueType at(const T& idx) const { return this->Data.at(idx); }

  template<typename OtherT>
  HostArray &operator=(const std::vector<OtherT> &v)
  { Data=(v.Data); return *this;}

  template<typename OtherT>
  HostArray &operator=(const HostArray<OtherT> &v)
  { Data=(v.Data); Name=v.name();return *this;}

  HostArray &operator=(const HostArray &v)
  { Data=(v.Data); Name=v.name(); return *this; }


protected:
  std::string Name;
  Parent Data;
};


typedef dax::HostArray<dax::Id> IdArray;
typedef dax::HostArray<dax::Scalar> ScalarArray;
typedef dax::HostArray<dax::Vector3> Vector3Array;

typedef dax::HostArray<dax::Vector3> Coordinates;


//typedef boost::shared_ptr< dax::Array<ValueType,StorageType> > ArrayPtr;
//typedef boost::weak_ptr< dax::Array<ValueType,StorageType> > ArrayWeakPtr;
}
#endif // DAXARRAY_H
