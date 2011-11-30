#ifndef WORKLETS_H
#define WORKLETS_H

#include <dax/internal/ExportMacros.h>
#include <dax/Types.h>

#include <math.h>
#include <vector>
#include <algorithm>

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

DAX_EXEC_CONT_EXPORT void CellGradient(dax::Scalar inScalar, dax::Vector3 inVec, dax::Scalar& outValue)
{
  outValue = (inScalar * inVec.x) +
             (inScalar * inVec.y) +
             (inScalar * inVec.z);
}
}

namespace worklets
{
struct Cosine
{
  enum {NumInputs = 1,
        NumOutputs = 1};

  template<typename T, typename U>
  DAX_EXEC_CONT_EXPORT void operator()(const T& in, U&out)
  {
    //passed down the port vectors as in and out
    //might need iterators here
    for(int i=0; i < in[0]->size(); i++)
      {
      worklet_functions::Cosine(in[0]->at(i),out[0].set(i));
      }
  }

  std::string name() const { return "Cosine"; }
};

struct Sine
{
  enum {NumInputs = 1,
        NumOutputs = 1};

  template<typename T, typename U>
  DAX_EXEC_CONT_EXPORT void operator()(const T& in, U&out)
  {
    //passed down the port vectors as in and out
    //might need iterators here
    for(int i=0; i < in[0]->size(); i++)
      {
      worklet_functions::Sine(in[0]->at(i),out[0].set(i));
      }
  }

  std::string name() const { return "Sine"; }
};

struct Square
{
  enum {NumInputs = 1,
        NumOutputs = 1};

  template<typename T, typename U>
  DAX_EXEC_CONT_EXPORT void operator()(const T& in, U&out)
  {
    //passed down the port vectors as in and out
    //might need iterators here
    for(int i=0; i < in[0]->size(); i++)
      {
      worklet_functions::Square(in[0]->at(i),out[0].set(i));
      }
  }

  std::string name() const { return "Square"; }
};

struct Elevation
{
  enum {NumInputs = 1,
        NumOutputs = 1};

  template<typename T, typename U>
  DAX_EXEC_CONT_EXPORT void operator()(const T& in, U&out)
  {

  }

  std::string name() const { return "Elevation"; }
};

struct CellGradient
{
  enum {NumInputs = 2,
        NumOutputs = 1};

  template<typename T, typename U>
  DAX_EXEC_CONT_EXPORT void operator()(const T& in, U&out)
  {

  }

  std::string name() const { return "CellGradient"; }
};

struct PointToCell
{
  enum {NumInputs = 1,
        NumOutputs = 1};

  template<typename T, typename U>
  DAX_EXEC_CONT_EXPORT void operator()(const T& in, U&out)
  {

  }

  std::string name() const { return "PointToCell"; }
};

struct CellToPoint
{
  enum {NumInputs = 1,
        NumOutputs = 1};

  template<typename T, typename U>
  DAX_EXEC_CONT_EXPORT void operator()(const T& in, U&out)
  {

  }

  std::string name() const { return "CellToPoint"; }
};
}

#endif // WORKLETS_H
