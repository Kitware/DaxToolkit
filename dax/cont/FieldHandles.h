#ifndef __dax_cont_FieldHandles_h
#define __dax_cont_FieldHandles_h

#include <string>
#include <dax/Types.h>

#include <boost/shared_ptr.hpp>
#include <dax/cont/internal/ArrayContainer.h>

namespace dax {
namespace cont {


template<typename T>
class pointFieldHandle
{
public:
  typedef T ValueType;


  pointFieldHandle(const std::string &name):
    Name(name)
    {

    }

  template<typename G>
  dax::Id static fieldSize(const G& g)
    {
    return g.numPoints();
    }

  template<typename G>
  void associate(G &g)
    {
    g.getFieldsPoint().addArray(this->Name,this->Container);
    }

  template<typename U>
  void setExecutionArray(U array)
    {
    this->Container.setExecution(array);
    }

  template<typename U>
  void setControlArray(U array)
    {
    this->Container.setControl(array);
    }

private:
  dax::cont::internal::ArrayContainer<ValueType> Container;
  const std::string Name;

};

template<typename T>
class cellFieldHandle
{
public:
  typedef T ValueType;

  cellFieldHandle(const std::string &name):
    Name(name)
    {
    }

  template<typename G>
  dax::Id static fieldSize(const G& g)
    {
    return g.numCells();
    }

  template<typename G>
  void associate(G &g)
    {
    g.getFieldsCell().addArray(this->Name,this->Container);
    }

  template<typename U>
  void setExecutionArray(U array)
    {
    this->Container.setExecution(array);
    }

  template<typename U>
  void setControlArray(U array)
    {
    this->Container.setControl(array);
    }

private:
  dax::cont::internal::ArrayContainer<ValueType> Container;
  const std::string Name;
};

} }


#endif // __dax_cont_FieldHandles_h
