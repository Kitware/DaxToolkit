#ifndef __dax_cont_Worklets_h
#define __dax_cont_Worklets_h


#include <dax/cont/internal/ArrayContainer.h>

namespace dax {
namespace cuda {
namespace cont {
namespace worklet {

//forward declare these.
//we need a factory instead that creates the correct worklet
class Elevation;
class Square;
class Sine;
class Cosine;
class CellGradient;
} } } }

namespace dax {
namespace cont {
namespace worklets {

class Elevation
{
public:
  template<typename G, typename T, typename U>
  Elevation(G &g, T& in, U out)
  {    
    dax::cuda::cont::worklet::Elevation()(g,in,out);
    out.associate(g);
  }
};

class Square
{
public:
  template<typename G, typename T, typename U>
  Square(G &g, T& in, U out)
  {
    dax::cuda::cont::worklet::Square()(g,in,out);
    out.associate(g);
  }

};

class Sine
{
public:
  template<typename G, typename T, typename U>
  Sine(G &g, T& in, U out)
  {
    dax::cuda::cont::worklet::Sine()(g,in,out);
    out.associate(g);
  }

};

class Cosine
{
public:
  template<typename G, typename T, typename U>
  Cosine(G &g, T& in, U out)
  {
    dax::cuda::cont::worklet::Cosine()(g,in,out);
    out.associate(g);
  }

};

class CellGradient
{

public:
  template<typename G, typename T, typename T2, typename U>
  CellGradient(G &g, T& in, T2& in2, U out)
  {
    dax::cuda::cont::worklet::CellGradient()(g,in,in2,out);
    out.associate(g);
  }
};


} } }


#endif // __dax_cont_Worklets_h
