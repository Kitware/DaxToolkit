#ifndef DAXARRAY_H
#define DAXARRAY_H

#include <vector>
#include <string>

#include "daxTypes.h"

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

class BaseCoordinates : public BaseArray
{
public:
  typedef dax::Vector3 OutputType;
  typedef dax::Vector3 ValueType;

  virtual ~BaseCoordinates(){}
  virtual std::size_t size() const=0;
  virtual dax::Vector3 operator [](const std::size_t& idx) const=0;
  virtual dax::Vector3 at(const std::size_t& idx) const=0;

  virtual std::string name() const { return "coords"; }
};

} }

namespace dax {

template<typename Type>
class HostArray :
    public dax::internal::BaseArray,
    public std::vector<Type>

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
    :Parent(v) {}

  HostArray &operator=(const HostArray &v)
  { Parent::operator=(v); return *this; }

  template<typename OtherT>
  HostArray(const HostArray<OtherT> &v)
    :Parent(v) {}

  template<typename OtherT>
  HostArray(const std::vector<OtherT> &v)
    :Parent(v) {}

  template<typename InputIterator>
  HostArray(InputIterator first, InputIterator last)
    :Parent(first, last) {}

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
  { Parent::operator=(v); return *this;}

  template<typename OtherT>
  HostArray &operator=(const HostArray<OtherT> &v)
  { Parent::operator=(v); return *this;}


protected:
  std::string Name;
  Parent Data;
};

typedef dax::HostArray<dax::Id> IdArray;
typedef dax::HostArray<dax::Scalar> ScalarArray;
typedef dax::HostArray<dax::Vector3> Vector3Array;

//I do not like this,
//I do not like
//Coordinates.

//I would not like them
//on the Heap or Stack.
//I would not like them
//anywhere.
//I do not like
//Coordinates.
//I do not like them


//I need to create a device version of coordinates
//the facade really sucks
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


void ConvertCoordinatesToArray(const dax::internal::BaseCoordinates *baseC,
               dax::Vector3Array& result)
{
  //I hate this, we really need a device agnostic
  //mapping of coordinate arrays
  std::size_t size = baseC->size();
  result.reserve(size);
  for(std::size_t i=0; i < size; ++i)
    {
    result.push_back((*baseC)[i]);
    }

  result.setName(baseC->name());

}


//typedef boost::shared_ptr< dax::Array<ValueType,StorageType> > ArrayPtr;
//typedef boost::weak_ptr< dax::Array<ValueType,StorageType> > ArrayWeakPtr;
}
#endif // DAXARRAY_H
