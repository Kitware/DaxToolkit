#ifndef __dax_cont_FieldHandles_h
#define __dax_cont_FieldHandles_h

#include <string>
#include <dax/Types.h>
#include <dax/cont/Array.h>

namespace dax {
namespace cont {


template<typename T>
class pointFieldHandle
{
public:
  typedef T DataType;
  dax::cont::ArrayPtr<DataType> Array;


  pointFieldHandle(const std::string &name):
    Name(name), ToDelete(true)
    {
    this->Array = dax::cont::ArrayPtr<DataType>(new dax::cont::Array<DataType>());
    }

  template<typename G>
  dax::Id static fieldSize(const G& g)
    {
    return g.numPoints();
    }

  template<typename G>
  void associate(G &g)
    {
    this->Array->resize(pointFieldHandle::fieldSize(g));
    g.getFieldsPoint().addArray(this->Name,this->Array);
    this->ToDelete = false;
    }

  dax::cont::ArrayPtr<T> array()
    {
    return this->Array;
    }

private:
  bool ToDelete;
  const std::string Name;

};

template<typename T>
class cellFieldHandle
{
public:
  typedef T DataType;
  dax::cont::ArrayPtr<DataType> Array;

  cellFieldHandle(const std::string &name):
    Name(name), ToDelete(true)
    {
    this->Array = dax::cont::ArrayPtr<DataType>(new dax::cont::Array<DataType>());
    }

  template<typename G>
  dax::Id static fieldSize(const G& g)
    {
    return g.numCells();
    }

  template<typename G>
  void associate(G &g)
    {
    this->Array->resize(cellFieldHandle::fieldSize(g));
    g.getFieldsCell().addArray(this->Name,this->Array);
    this->ToDelete = false;
    }

  dax::cont::ArrayPtr<T> array()
    {
    return this->Array;
    }


private:
  bool ToDelete;
  const std::string Name;
};

} }


#endif // __dax_cont_FieldHandles_h
