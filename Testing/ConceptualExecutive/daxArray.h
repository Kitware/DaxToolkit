#ifndef DAXARRAY_H
#define DAXARRAY_H

#include <vector>
#include <string>

#include "daxTypes.h"

namespace dax { namespace internal {

class BaseArray
{
public:
    virtual ~BaseArray(){}
    virtual std::size_t size() const=0;
    virtual std::string name() const=0;
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

//expose the storage type so that
//we can easily convert the underlying array
//from an std::vector to thrust vector
template<typename valueType,
          typename storageType=std::vector<valueType> >
class Array : public dax::internal::BaseArray
{
public:
  typedef valueType ValueType;
  typedef storageType StorageType;

  typedef typename StorageType::const_iterator const_iterator;
  typedef typename StorageType::iterator iterator;

  Array(const std::string& name, const int& reserveSize=0):
  Name(name)
  {
  this->Data.reserve(reserveSize);
  }

  virtual ~Array(){}

  void setName(const std::string &name) { Name = name; }
  virtual std::string name() const { return Name; }

  std::size_t size() const { return this->Data.size(); }
  std::size_t capacity() const { return this->Data.capacity(); }

  void reserve(const std::size_t& sz) { this->Data.reserve(sz); }

  template <typename ValueType>
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

typedef dax::Array<dax::Id,std::vector<dax::Id> > IdArray;
typedef dax::Array<dax::Scalar,std::vector<dax::Scalar> > ScalarArray;
typedef dax::Array<dax::Vector3,std::vector<dax::Vector3> > Vector3Array;


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
    return this->Data->at(idx);
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
