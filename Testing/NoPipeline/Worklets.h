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
    Name(name)
    {
    this->Array = new dax::cont::Array<T>();
    this->Array->setName(this->Name);
    }

  template<typename G>
  void associate(G &g)
    {
    g.addPointField(this->Array);
    }

  dax::cont::Array<T>& array()
    {
    return *this->Array;
    }

private:
  const std::string Name;

};

template<typename T>
class cellFieldHandle
{
public:
  typedef T DataType;
  dax::cont::Array<T> *Array;

  cellFieldHandle(const std::string &name):
    Name(name)
    {
    this->Array = new dax::cont::Array<T>();
    this->Array->setName(this->Name);
    }

  template<typename G>
  void associate(G &g)
    {
    g.addCellField(this->Array);
    }

  dax::cont::Array<T>& array()
    {
    return *this->Array;
    }


private:
  const std::string Name;
};

}

namespace worklets {

class Elevation
{
public:
  template<typename G, typename T, typename U>
  Elevation(G &g, T& in, U out)
  {
    workletProxies::Elevation()(g,in,out);
  }
};

class Square
{
public:
  template<typename G, typename T, typename U>
  Square(G &g, T& in, U out)
  {
    workletProxies::Square()(g,in,out);
  }

};

class Sine
{
public:
  template<typename G, typename T, typename U>
  Sine(G &g, T& in, U out)
  {
    //workletProxies::Elevation()(g,in,out);
  }

};

class Cosine
{
public:
  template<typename G, typename T, typename U>
  Cosine(G &g, T& in, U out)
  {
    //workletProxies::Elevation()(g,in,out);
  }

};

class CellGradient
{

public:
  template<typename G, typename T, typename T2, typename U>
  CellGradient(G &g, T& in, T2& in2, U out)
  {
    //workletProxies::Elevation()(g,in,out);
  }
};


}


#endif // NO_PIPELINE_WORKLETS_H
