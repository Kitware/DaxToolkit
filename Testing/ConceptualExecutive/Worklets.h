#ifndef WORKLETS_H
#define WORKLETS_H

#include <dax/internal/ExportMacros.h>

#include <math.h>
#include "daxTypes.h"
#include <thrust/device_vector.h>

//------------------------------------------------------------------------------
template <typename Worklet>
struct execute
{
  template<typename A, typename B>
  void operator()(const A& input, B* output)
  {
    operator()(input,*output);
  }
  template<typename A, typename B>
  void operator()(const A* input, B& output)
  {
    operator()(*input,output);
  }
  template<typename A, typename B>
  void operator()(const A* input, B* output)
  {
    operator()(*input,*output);
  }
  template<typename A, typename B>
  void operator()(const A &input, B &output)
  {
    Worklet work;
    output.resize(input.size());
    for(int i=0; i < input.size(); i++)
      {
      output[(work.run(input[i]))];
      }
  }
};

namespace worklet_functions
{
DAX_EXEC_CONT_EXPORT void Cosine(dax::Scalar inValue, dax::Scalar &outValue)
{
  outValue = cosf(inValue);
}

DAX_EXEC_CONT_EXPORT void Sine(dax::Scalar inValue, dax::Scalar &outValue)
{
  outValue = sinf(inValue);
}

DAX_EXEC_CONT_EXPORT void Square(dax::Scalar inValue, dax::Scalar &outValue)
{
  outValue = inValue * inValue;
}

DAX_EXEC_CONT_EXPORT void Elevation(dax::Vector3 inCoordinate, dax::Scalar& outValue)
{
  outValue = sqrtf( dax::dot(inCoordinate, inCoordinate) );
}
}

namespace worklets
{
struct Cosine
{
  typedef dax::Scalar InputType;
  typedef dax::Scalar OutputType;
  DAX_EXEC_CONT_EXPORT dax::Scalar run(dax::Scalar v)
  {
    worklet_functions::Cosine(v,v);
    return v;
  }
};

struct Sine
{
  typedef dax::Scalar InputType;
  typedef dax::Scalar OutputType;
  DAX_EXEC_CONT_EXPORT dax::Scalar run(dax::Scalar v)
  {
    worklet_functions::Sine(v,v);
    return v;
  }
};

struct Square
{
  typedef dax::Scalar InputType;
  typedef dax::Scalar OutputType;
  DAX_EXEC_CONT_EXPORT dax::Scalar run(dax::Scalar v)
  {
    worklet_functions::Square(v,v);
    return v;
  }
};

struct Elevation
{
  typedef dax::Vector3 InputType;
  typedef dax::Scalar OutputType;
  DAX_EXEC_CONT_EXPORT dax::Scalar run(dax::Vector3 v)
  {
    dax::Scalar result;
    worklet_functions::Elevation(v,result);
    return result;
  }
};

}

#endif // WORKLETS_H
