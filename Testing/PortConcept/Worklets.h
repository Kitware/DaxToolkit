#ifndef WORKLETS_H
#define WORKLETS_H

#include <dax/internal/ExportMacros.h>
#include <dax/Types.h>

#include <assert.h>
#include "StaticAssert.h"

#include <math.h>
#include <vector>
#include <algorithm>

#include <dax/cont/StructuredGrid.h>

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
  DAX_EXEC_CONT_EXPORT void operator()(const T& in, U& out)
  {
    //passed down the port vectors as in and out
    //might need iterators here
    for(int i=0; i < in[0].size(); i++)
      {
      worklet_functions::Cosine(in[0].at(i),out[0].set(i));
      }
  }

  std::string name() const { return "Cosine"; }
};

struct Sine
{
  enum {NumInputs = 1,
        NumOutputs = 1};

  template<typename T, typename U>
  DAX_EXEC_CONT_EXPORT void operator()(const T& in, U& out)
  {
    //passed down the port vectors as in and out
    //might need iterators here
    for(int i=0; i < in[0].size(); i++)
      {
      worklet_functions::Sine(in[0].at(i),out[0].set(i));
      }
  }

  std::string name() const { return "Sine"; }
};

struct Square
{
  enum {NumInputs = 1,
        NumOutputs = 1};

  template<typename T, typename U>
  DAX_EXEC_CONT_EXPORT void operator()(const T& in, U& out)
  {
    //passed down the port vectors as in and out
    //might need iterators here
    for(int i=0; i < in[0].size(); i++)
      {
      worklet_functions::Square(in[0].at(i),out[0].set(i));
      }
  }

  std::string name() const { return "Square"; }
};

struct Elevation
{
  enum {NumInputs = 1,
        NumOutputs = 1};

  template<typename T, typename U>
  DAX_EXEC_CONT_EXPORT void operator()(const T& in, U& out)
  {

  }

  std::string name() const { return "Elevation"; }
};

struct CellGradient
{
  enum {NumInputs = 2,
        NumOutputs = 1};

  template<typename T, typename U>
  DAX_EXEC_CONT_EXPORT void operator()(const T& in, U& out)
  {
    //passed down the port vectors as in and out
    //might need iterators here
    for(int i=0; i < in[0].size(); i++)
      {
      worklet_functions::Square(in[0]->at(i),
                                in[1]->at(i),
                                out[0].set(i));
      }
  }

  std::string name() const { return "CellGradient"; }
};

struct PointToCell
{
  enum {NumInputs = 1,
        NumOutputs = 1};

  template<typename T, typename U>
  DAX_EXEC_CONT_EXPORT void operator()(const T& in, U& out)
  {

  }

  std::string name() const { return "PointToCell"; }
};

struct CellToPoint
{
  enum {NumInputs = 1,
        NumOutputs = 1};

  template<typename T, typename U>
  DAX_EXEC_CONT_EXPORT void operator()(const T& in, U& out)
  {

  }

  std::string name() const { return "CellToPoint"; }
};

struct ChangeTopology
{
  enum {NumInputs = 1,
        NumOutputs = 1};

  //parameter includes all the input ports so
  //it can decide based on that information
  template<typename T>
  dax::cont::StructuredGrid* requestDataSet(int portId, const T&)
  {

    //this will construct the new data set for this
    //proxy. It should only be called once for each
    //output port
    assert(portId>=0 && portId<NumOutputs);
    return new dax::cont::StructuredGrid();
  }

  template<typename T, typename U>
  DAX_EXEC_CONT_EXPORT void operator()(const T& in, U& out)
  {

    std::cout << "getting current topology" << std::endl;

    dax::cont::DataSet* dsIn = in[0].dataSet();
    dax::cont::DataSet* dsOut = out[0].dataSet();

    dax::cont::StructuredGrid* sgIn =
        dynamic_cast<dax::cont::StructuredGrid*>(dsIn);
    dax::cont::StructuredGrid* sgOut =
        dynamic_cast<dax::cont::StructuredGrid*>(dsOut);
    if(sgIn && sgOut)
      {
      std::cout << "change topology" << std::endl;
      sgOut->Origin = sgIn->Origin;
      sgOut->Spacing = sgIn->Spacing;
      sgOut->Extent.Max = dax::make_Id3(sgIn->Extent.Max.x/2, sgIn->Extent.Max.y/2, sgIn->Extent.Max.z/2);
      sgOut->Extent.Min = sgIn->Extent.Min;
      }
  }

  std::string name() const { return "ChangeTopology"; }
};
}

#endif // WORKLETS_H
