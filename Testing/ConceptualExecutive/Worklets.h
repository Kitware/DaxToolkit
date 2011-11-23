#ifndef WORKLETS_H
#define WORKLETS_H

#include <math.h>
#include "daxTypes.h"

namespace worklet_functions
{
void Cosine(dax::Scalar inValue, dax::Scalar &outValue)
{
  outValue = cosf(inValue);
}

void Sine(dax::Scalar inValue, dax::Scalar &outValue)
{
  outValue = sinf(inValue);
}

void Square(dax::Scalar inValue, dax::Scalar &outValue)
{
  outValue = inValue * inValue;
}

void Elevation(dax::Vector3 inCoordinate, dax::Scalar& outValue)
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
  dax::Scalar run(dax::Scalar v)
  {
    worklet_functions::Cosine(v,v);
    return v;
  }
};

struct Sine
{
  typedef dax::Scalar InputType;
  typedef dax::Scalar OutputType;
  dax::Scalar run(dax::Scalar v)
  {
    worklet_functions::Sine(v,v);
    return v;
  }
};

struct Square
{
  typedef dax::Scalar InputType;
  typedef dax::Scalar OutputType;
  dax::Scalar run(dax::Scalar v)
  {
    worklet_functions::Square(v,v);
    return v;
  }
};

struct Elevation
{
  typedef dax::Vector3 InputType;
  typedef dax::Scalar OutputType;
  dax::Scalar run(dax::Vector3 v)
  {
    dax::Scalar result;
    worklet_functions::Elevation(v,result);
    return result;
  }
};

}

#endif // WORKLETS_H
