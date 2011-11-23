#ifndef WORKLETS_H
#define WORKLETS_H

#include <dax/internal/ExportMacros.h>

#include <math.h>
#include <algorithm>
#include "daxTypes.h"

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
  DAX_EXEC_CONT_EXPORT dax::Scalar operator()(dax::Scalar v)
  {
    worklet_functions::Cosine(v,v);
    return v;
  }
};

struct Sine
{
  typedef dax::Scalar InputType;
  typedef dax::Scalar OutputType;
  DAX_EXEC_CONT_EXPORT dax::Scalar operator()(dax::Scalar v)
  {
    worklet_functions::Sine(v,v);
    return v;
  }
};

struct Square
{
  typedef dax::Scalar InputType;
  typedef dax::Scalar OutputType;
  DAX_EXEC_CONT_EXPORT dax::Scalar operator()(dax::Scalar v)
  {
    worklet_functions::Square(v,v);
    return v;
  }
};

struct Elevation
{
  typedef dax::Vector3 InputType;
  typedef dax::Scalar OutputType;
  DAX_EXEC_CONT_EXPORT dax::Scalar operator()(dax::Vector3 v)
  {
    dax::Scalar result;
    worklet_functions::Elevation(v,result);
    return result;
  }
};
}

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
    output.resize(input.size());
    for(int i=0; i < input.size(); i++)
      {
      output[i]=(Worklet()(input[i]));
      }
  }
};

//------------------------------------------------------------------------------
template <typename DataType,
          typename InputType,
          typename OutputType>
struct cell_execute
{
  int index;
  const DataType* Data;

  cell_execute(const DataType *data):
    index(0), Data(data)
  {

  }


  DAX_EXEC_CONT_EXPORT
  void operator()(OutputType& value)
  {
    value = worklets::Cosine()(
            worklets::Square()(
            worklets::Sine()(
            worklets::Elevation()((*Data)[index++]))));
  }
};


template <typename T, typename U>
void executePipeline(T *points, U &result)
{
  result.resize(points->size());
  cell_execute<T, typename T::value_type, typename U::value_type> ce(points);
  std::for_each(result.begin(),result.end(),ce);
}


#endif // WORKLETS_H
