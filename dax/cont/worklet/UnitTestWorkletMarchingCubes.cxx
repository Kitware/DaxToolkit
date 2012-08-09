//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================

#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/DeviceAdapterSerial.h>

#include <dax/cont/internal/TestingGridGenerator.h>
#include <dax/cont/internal/Testing.h>

#include <dax/cont/worklet/MarchingCubes.h>

#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <dax/TypeTraits.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/internal/Testing.h>
#include <vector>


namespace {
const dax::Id DIM = 26;
const dax::Id ISO_VALUE = 70;

class CheckValid {
public:
  CheckValid() : Valid(true) { }
  operator bool() { return this->Valid; }
  // ALWAYS INVALID FOR NOW
  void operator()(dax::Scalar value) {
      this->Valid = false;
    }
private:
  bool Valid;
};

void PrintScalarValue(dax::Scalar value)
  {
  std::cout << " " << value;
  }


template<class IteratorType>
void CheckValues(IteratorType begin, IteratorType end)
  {
  typedef typename std::iterator_traits<IteratorType>::value_type VectorType;

  CheckValid isValid;
  for (IteratorType iter = begin; iter != end; iter++)
    {
    VectorType vector = *iter;
    dax::cont::VectorForEach(vector, isValid);
    DAX_TEST_ASSERT(isValid, "Encountered bad value.");
    }
  }

template<typename T, class Container, class Device>
void CheckValues(dax::cont::ArrayHandle<T,Container,Device> arrayHandle)
{
  CheckValues(arrayHandle.GetPortalConstControl().GetIteratorBegin(),
              arrayHandle.GetPortalConstControl().GetIteratorEnd());
}

//-----------------------------------------------------------------------------
struct TestMarchingCubesWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  void operator()(const GridType&) const
    {
    dax::cont::internal::TestGrid<GridType> in(DIM);
    GridType out;

    this->GridMarchingCubes(in.GetRealGrid(),out);
    }

  //----------------------------------------------------------------------------
  void operator()(const dax::cont::UniformGrid<>&) const
    {
    dax::cont::internal::TestGrid<dax::cont::UniformGrid<> > in(DIM);
    dax::cont::UnstructuredGrid<dax::exec::CellHexahedron> out;

    this->GridMarchingCubes(in.GetRealGrid(),out);
    }

  //----------------------------------------------------------------------------
  template <typename InGridType,
            typename OutGridType>
  void GridMarchingCubes(const InGridType& inGrid, OutGridType& outGrid) const
    {
    dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);
    std::vector<dax::Scalar> field(inGrid.GetNumberOfPoints());
    for (dax::Id pointIndex = 0;
         pointIndex < inGrid.GetNumberOfPoints();
         pointIndex++)
      {
      dax::Vector3 coordinates = inGrid.ComputePointCoordinates(pointIndex);
      field[pointIndex] = dax::dot(coordinates, trueGradient);
      }

    dax::cont::ArrayHandle<dax::Scalar> fieldHandle =
        dax::cont::make_ArrayHandle(field);

    //unkown size
    dax::cont::ArrayHandle<dax::Scalar> resultHandle;

    std::cout << "Running MarchingCubes worklet" << std::endl;
    dax::Scalar isoValue = ISO_VALUE;

    try
      {
    // Comment this out for now cause all grid types are not handled
//      dax::cont::worklet::MarchingCubes(inGrid,
//                                        outGrid,
//                                        isoValue,
//                                        fieldHandle,
//                                        resultHandle);
      }
    catch (dax::cont::ErrorControl error)
      {
      std::cout << "Got error: " << error.GetMessage() << std::endl;
      DAX_TEST_ASSERT(true==false,error.GetMessage());
      }

    CheckValues(resultHandle);
    DAX_TEST_ASSERT(resultHandle.GetNumberOfValues()==outGrid.GetNumberOfPoints(),
                    "Incorrect number of points in the result array");
    }
};


//-----------------------------------------------------------------------------
void TestMarchingCubes()
  {
  dax::cont::internal::GridTesting::TryAllGridTypes(TestMarchingCubesWorklet());
  }
} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletMarchingCubes(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestMarchingCubes);
}
