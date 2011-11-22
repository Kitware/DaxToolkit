#ifndef WORKLETS_H
#define WORKLETS_H

#include <math.h>

namespace worklets
{
void Cosine(float inValue, float &outValue)
{
  outValue = cosf(inValue);
}

void Sine(float inValue, float &outValue)
{
  outValue = sinf(inValue);
}

void Square(float inValue, float &outValue)
{
  outValue = inValue * inValue;
}
}

namespace sWorklets
{

struct Cosine
{
  float run(float v)
  {
    worklets::Cosine(v,v);
    return v;
  }
};

struct Sine
{
  float run(float v)
  {
    worklets::Sine(v,v);
    return v;
  }
};

struct Square
{
  float run(float v)
  {
    worklets::Square(v,v);
    return v;
  }
};

}

#endif // WORKLETS_H
