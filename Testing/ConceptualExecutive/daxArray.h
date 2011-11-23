#ifndef DAXARRAY_H
#define DAXARRAY_H

#include <vector>
#include <string>

#include <dax/internal/ExportMacros.h>

#include "daxTypes.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace dax { namespace internal {

class BaseArray
{
public:
    virtual ~BaseArray(const std::string &n):
    Name(n)
    {
    }
    virtual std::string name() const=0;
protected:
    std::string Name;
  };

class BaseCoordinates : public BaseArray
{
public:
  typedef dax::Vector3 OutputType;
  virtual ~BaseCoordinates(){}
  virtual std::size_t size() const=0;
  virtual dax::Vector3 operator [](const std::size_t& idx) const=0;
  virtual dax::Vector3 at(const std::size_t& idx) const=0;

  virtual std::string name() const { return "coords"; }
};
} }

namespace dax {

template<typename valueType>
class HostArray :
    public dax::internal::BaseArray,
    public thrust::host_vector<valueType>

{
private:
  typedef thrust::host_vector<valueType> Parent;

public:
  HostArray(const std::string& name):
    BaseArray(name),Parent()
    {}

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

  template<typename T>
  ValueType operator [](const T& idx) const { return this->Data[idx]; }

  template<typename T>
  ValueType& operator [](const T& idx) { return this->Data[idx]; }

  template<typename T>
  ValueType at(const T& idx) const { return this->Data.at(idx); }

protected:
  std::string Name;
  StorageType Data;
};

template<typename T, typename Alloc = thrust::device_malloc_allocator<T> >
class DeviceArray : public thrust::device_vector<T,Alloc>
{
private:
  typedef thrust::device_vector<T,Alloc> Parent;

  /*! Assign operator copies from an exemplar <tt>std::vector</tt>.
   *  \param v The <tt>std::vector</tt> to copy.
   */
  template<typename OtherT>
  __host__
  DeviceArray &operator=(const dax::HostArray<OtherT> &v)
  { Parent::operator=(v.Data); return *this;}
};

typedef dax::HostArray<dax::Id,thrust::host_vector<dax::Id> > IdArray;
typedef dax::HostArray<dax::Scalar,thrust::host_vector<dax::Scalar> > ScalarArray;
typedef dax::HostArray<dax::Vector3,thrust::host_vector<dax::Vector3> > Vector3Array;

typedef dax::DeviceArray<dax::Id> DeviceIdArray;
typedef dax::DeviceArray<dax::Scalar> DeviceScalarArray;
typedef dax::DeviceArray<dax::Vector3> Vector3Array;



template <typename realArray>
class Coordinates : public dax::internal::BaseCoordinates
{
public:
  Coordinates(realArray* ra)
  {
    this->RealArrayPtr = ra;
  }

  virtual ~Coordinates()
  {
    RealArrayPtr = NULL;
  }

  virtual std::size_t size() const { return this->RealArrayPtr->size(); }

  virtual dax::Vector3 operator [](const std::size_t& idx) const
  { return this->RealArrayPtr[idx]; }

  virtual dax::Vector3 at(const std::size_t& idx) const
  {
    return this->RealArrayPtr->at(idx);
  }

protected:
  realArray* RealArrayPtr;
};

template <typename dataSet>
class ComputedCoordinates : public dax::internal::BaseCoordinates
{
public:
  typedef dataSet InputType;
  ComputedCoordinates(const InputType* ds):
    Data(ds)
  {
  }

  virtual ~ComputedCoordinates()
  {
    Data = NULL;
  }

  virtual std::size_t size() const
  {
    return this->Data->numPoints();
  }

  virtual dax::Vector3 operator [](const std::size_t& idx) const
  {
    return this->Data->computePointCoordinate(idx);
  }

  virtual dax::Vector3 at(const std::size_t& idx) const
  {
    return this->Data->computePointCoordinate(idx);
  }
protected:
  const InputType *Data;
};


//typedef boost::shared_ptr< dax::Array<ValueType,StorageType> > ArrayPtr;
//typedef boost::weak_ptr< dax::Array<ValueType,StorageType> > ArrayWeakPtr;
}
#endif // DAXARRAY_H
