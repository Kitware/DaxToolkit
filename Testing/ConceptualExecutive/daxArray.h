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
    virtual ~BaseArray(){}
    virtual std::string name() const=0;
  };

class BaseCoordinates : public BaseArray
{
public:
  typedef dax::Vector3 OutputType;
  typedef dax::Vector3 value_type;
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
  typedef typename Parent::size_type  size_type;
  typedef typename Parent::value_type value_type;

  HostArray(const std::string& name):
    BaseArray(),Parent(),
    Name(name)
    {}

  virtual ~HostArray(){}

  virtual std::string name() const { return Name; }

protected:
    const std::string Name;
};

template<typename valueType>
class DeviceArray : public thrust::device_vector<valueType>
{
private:
  typedef thrust::device_vector<valueType> Parent;

public:
  typedef typename Parent::size_type  size_type;
  typedef typename Parent::value_type value_type;

  __host__
  DeviceArray(void)
    :Parent() {}

  __host__
  explicit DeviceArray(size_type n, const value_type &value = value_type())
    :Parent(n,value) {}

  __host__
  DeviceArray(const DeviceArray &v)
    :Parent(v) {}
};

typedef dax::HostArray<dax::Id> IdArray;
typedef dax::HostArray<dax::Scalar> ScalarArray;
typedef dax::HostArray<dax::Vector3> Vector3Array;

typedef dax::DeviceArray<dax::Id> DeviceIdArray;
typedef dax::DeviceArray<dax::Scalar> DeviceScalarArray;
typedef dax::DeviceArray<dax::Vector3> DeviceVector3Array;



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
