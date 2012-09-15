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
#include <stdio.h>
#include <iostream>
#include "Timer.h"

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/worklet/CellGradient.h>
#include <dax/cont/worklet/Cosine.h>
#include <dax/cont/worklet/Magnitude.h>
#include <dax/cont/worklet/Sine.h>
#include <dax/cont/worklet/Square.h>

#include <vector>

#define MAKE_STRING2(x) #x
#define MAKE_STRING1(x) MAKE_STRING2(x)
#define DEVICE_ADAPTER MAKE_STRING1(DAX_DEFAULT_DEVICE_ADAPTER_TAG)

namespace
{

class CheckValid {
public:
  CheckValid() : Valid(true) { }
  operator bool() { return this->Valid; }
  void operator()(dax::Scalar value) {
    if ((value < -1) || (value > 2)) { this->Valid = false; }
  }
private:
  bool Valid;
};

void PrintScalarValue(dax::Scalar value)
{
  std::cout << " " << value;
}

template<class IteratorType>
void PrintCheckValues(IteratorType begin, IteratorType end)
{
  typedef typename std::iterator_traits<IteratorType>::value_type VectorType;

  dax::Id index = 0;
  for (IteratorType iter = begin; iter != end; iter++)
    {
    VectorType vector = *iter;
    if (index < 20)
      {
      std::cout << index << ":";
      dax::cont::VectorForEach(vector, PrintScalarValue);
      std::cout << std::endl;
      }
    else
      {
      CheckValid isValid;
      dax::cont::VectorForEach(vector, isValid);
      if (!isValid)
        {
        std::cout << "*** Encountered bad value." << std::endl;
        std::cout << index << ":";
        dax::cont::VectorForEach(vector, PrintScalarValue);
        std::cout << std::endl;
        exit(1);
        }
      }

    index++;
    }
}

template<typename T, class Container, class Device>
void PrintCheckValues(const dax::cont::ArrayHandle<T,Container,Device> &array)
{
  PrintCheckValues(array.GetPortalConstControl().GetIteratorBegin(),
                   array.GetPortalConstControl().GetIteratorEnd());
}

void PrintResults(int pipeline, double time)
{
  std::cout << "Elapsed time: " << time << " seconds." << std::endl;
  std::cout << "CSV," DEVICE_ADAPTER ","
            << pipeline << "," << time << std::endl;
}

void RunPipeline1(const dax::cont::UniformGrid<> &grid)
{
  std::cout << "Running pipeline 1: Magnitude -> Gradient" << std::endl;

  dax::cont::ArrayHandle<dax::Scalar> intermediate1;

  dax::cont::ArrayHandle<dax::Vector3> results;

  Timer timer;
  dax::cont::Schedule<> schedule;
  schedule(dax::worklet::Magnitude(),
        grid.GetPointCoordinates(),
        intermediate1);
  schedule(dax::worklet::CellGradient(),grid,
                                   grid.GetPointCoordinates(),
                                   intermediate1,
                                   results);
  double time = timer.elapsed();

  PrintCheckValues(results);
  PrintResults(1, time);
}

void RunPipeline2(const dax::cont::UniformGrid<> &grid)
{
  std::cout << "Running pipeline 2: Magnitude->Gradient->Sine->Square->Cosine"
            << std::endl;

  dax::cont::ArrayHandle<dax::Scalar> intermediate1;
  dax::cont::ArrayHandle<dax::Vector3> intermediate2;
  dax::cont::ArrayHandle<dax::Vector3> intermediate3;

  dax::cont::ArrayHandle<dax::Vector3> results;

  Timer timer;
  dax::cont::Schedule<> schedule;
  schedule(dax::worklet::Magnitude(),
        grid.GetPointCoordinates(),
        intermediate1);

  schedule(dax::worklet::CellGradient(),grid,
           grid.GetPointCoordinates(),
           intermediate1,
           intermediate2);

  intermediate1.ReleaseResources();
  schedule(dax::worklet::Sine(),intermediate2, intermediate3);
  schedule(dax::worklet::Square(),intermediate3, intermediate2);
  intermediate3.ReleaseResources();
  schedule(dax::worklet::Cosine(),intermediate2, results);
  double time = timer.elapsed();

  PrintCheckValues(results);

  PrintResults(2, time);
}

void RunPipeline3(const dax::cont::UniformGrid<> &grid)
{
  std::cout << "Running pipeline 3: Magnitude -> Sine -> Square -> Cosine"
            << std::endl;

  dax::cont::ArrayHandle<dax::Scalar> intermediate1;
  dax::cont::ArrayHandle<dax::Scalar> intermediate2;

  dax::cont::ArrayHandle<dax::Scalar> results;

  Timer timer;
  dax::cont::Schedule<> schedule;
  schedule(dax::worklet::Magnitude(),
        grid.GetPointCoordinates(),
        intermediate1);
  schedule(dax::worklet::Sine(),intermediate1, intermediate2);
  schedule(dax::worklet::Square(),intermediate2, intermediate1);
  intermediate2.ReleaseResources();
  schedule(dax::worklet::Cosine(),intermediate1, results);
  double time = timer.elapsed();

  PrintCheckValues(results);

  PrintResults(3, time);
}

} // Anonymous namespace

