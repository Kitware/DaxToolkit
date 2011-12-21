#ifndef __dax_cont_Worklets_h
#define __dax_cont_Worklets_h


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
  Elevation(G &g, const T& in, U out)
  {
    out.associate(g);
    dax::cuda::cont::worklet::Elevation()(g,in,out);
  }
};

class Square
{
public:
  template<typename G, typename T, typename U>
  Square(G &g, const T& in, U out)
  {
    out.associate(g);
    dax::cuda::cont::worklet::Square()(g,in,out);
  }

};

class Sine
{
public:
  template<typename G, typename T, typename U>
  Sine(G &g, const T& in, U out)
  {
    out.associate(g);
    dax::cuda::cont::worklet::Sine()(g,in,out);
  }

};

class Cosine
{
public:
  template<typename G, typename T, typename U>
  Cosine(G &g, const T& in, U out)
  {
    out.associate(g);
    dax::cuda::cont::worklet::Cosine()(g,in,out);
  }

};

class CellGradient
{

public:
  template<typename G, typename T, typename T2, typename U>
  CellGradient(G &g, const T& in, const T2& in2, U out)
  {
    out.associate(g);
    dax::cuda::cont::worklet::CellGradient()(g,in,in2,out);
  }
};


} } }


#endif // __dax_cont_Worklets_h
