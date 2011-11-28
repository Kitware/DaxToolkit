#ifndef WORKLETS_H
#define WORKLETS_H

#include <dax/internal/ExportMacros.h>

#include <math.h>
#include <algorithm>

#include "daxTypes.h"
#include "daxArray.h"
#include "daxDeviceArray.h"

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
struct Cosine : public dax::worklets::BaseFieldWorklet<Cosine>
{
  typedef dax::Scalar InputType;
  typedef dax::Scalar OutputType;
  DAX_EXEC_CONT_EXPORT dax::Scalar operator()(dax::Scalar v)
  {
    worklet_functions::Cosine(v,v);
    return v;
  }

  std::string name() const { return "Cosine"; }
};

struct Sine : public dax::worklets::BaseFieldWorklet<Sine>
{
  typedef dax::Scalar InputType;
  typedef dax::Scalar OutputType;
  DAX_EXEC_CONT_EXPORT dax::Scalar operator()(dax::Scalar v)
  {
    worklet_functions::Sine(v,v);
    return v;
  }

  std::string name() const { return "Sine"; }
};

struct Square : public dax::worklets::BaseFieldWorklet<Square>
{
  typedef dax::Scalar InputType;
  typedef dax::Scalar OutputType;
  DAX_EXEC_CONT_EXPORT dax::Scalar operator()(dax::Scalar v)
  {
    worklet_functions::Square(v,v);
    return v;
  }

  std::string name() const { return "Square"; }
};

struct Elevation : public dax::worklets::BaseFieldWorklet<Elevation>
{
  typedef dax::Vector3 InputType;
  typedef dax::Scalar OutputType;
  DAX_EXEC_CONT_EXPORT dax::Scalar operator()(dax::Vector3 v)
  {
    dax::Scalar result;
    worklet_functions::Elevation(v,result);
    return result;
  }

  std::string name() const { return "Elevation"; }
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
  DAX_EXEC_CONT_EXPORT
  void operator()(const A &input, B &output)
  {
    for(int i=0; i < input.size(); i++)
      {
      output[i]=(Worklet()(input[i]));
      }
  }
};

//------------------------------------------------------------------------------
template <typename InputType,
          typename OutputType>
struct cell_execute
{
  DAX_EXEC_CONT_EXPORT
  OutputType operator()(const InputType& input)
  {
    return worklets::Cosine()(
            worklets::Square()(
            worklets::Sine()(
            worklets::Elevation()(input))));
  }
};


template <typename T, typename U>
void executeCellPipeline(const dax::HostArray<T> &points, dax::HostArray<U> &result)
{

  result.resize(points.size());
  cell_execute< typename dax::HostArray<T>::ValueType,
                typename dax::HostArray<U>::ValueType> functor;

  //currently thrust transform spits out warnings when running on the client
  //which means we can't compile
  std::cout << "start execution on the cpu" << std::endl;

  std::transform(points.begin(),points.end(),result.begin(),functor);

  std::cout << "finished execution on the cpu" << std::endl;
}

template <typename T, typename U>
void executeCellPipeline(const dax::DeviceArray<T> &points, dax::DeviceArray<U> &result)
{

  result.resize(points.size());
  cell_execute< typename dax::DeviceArray<T>::ValueType,
                typename dax::DeviceArray<U>::ValueType> functor;

  std::cout << "start execution on gpu" << std::endl;

  thrust::transform(points.begin(),points.end(),result.begin(),functor);

  std::cout << "finished execution on gpu" << std::endl;
}






#endif // WORKLETS_H
