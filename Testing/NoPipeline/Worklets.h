#ifndef NO_PIPELINE_WORKLETS_H
#define NO_PIPELINE_WORKLETS_H


#include "WorkletProxies.h"

namespace helpers
{

template<typename T>
class pointFieldHandle
{
public:
  typedef T DataType;
  dax::cont::Array<T> *Array;


  pointFieldHandle(const std::string &name):
    Name(name), ToDelete(true)
    {
    this->Array = new dax::cont::Array<T>();
    }

  virtual ~pointFieldHandle()
    {
    if(ToDelete)
      {
      delete this->Array;
      }
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

  dax::cont::Array<T>& array()
    {
    return *this->Array;
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
  dax::cont::Array<T> *Array;

  cellFieldHandle(const std::string &name):
    Name(name), ToDelete(true)
    {
    this->Array = new dax::cont::Array<T>();
    }

  virtual ~cellFieldHandle()
    {
    if(ToDelete)
      {
      delete this->Array;
      }
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

  dax::cont::Array<T>& array()
    {
    return *this->Array;
    }


private:
  bool ToDelete;
  const std::string Name;
};

}

namespace worklets {

class Elevation
{
public:
  template<typename G, typename T, typename U>
  Elevation(G &g, const T& in, U out)
  {
    out.associate(g);
    workletProxies::Elevation()(g,in,out);
  }
};

class Square
{
public:
  template<typename G, typename T, typename U>
  Square(G &g, const T& in, U out)
  {
    out.associate(g);
    workletProxies::Square()(g,in,out);
  }

};

class Sine
{
public:
  template<typename G, typename T, typename U>
  Sine(G &g, const T& in, U out)
  {
    out.associate(g);
    workletProxies::Sine()(g,in,out);
  }

};

class Cosine
{
public:
  template<typename G, typename T, typename U>
  Cosine(G &g, const T& in, U out)
  {
    out.associate(g);
    workletProxies::Cosine()(g,in,out);
  }

};

class CellGradient
{

public:
  template<typename G, typename T, typename T2, typename U>
  CellGradient(G &g, const T& in, const T2& in2, U out)
  {
    out.associate(g);
    workletProxies::CellGradient()(g,in,in2,out);
  }
};


}


#endif // NO_PIPELINE_WORKLETS_H
