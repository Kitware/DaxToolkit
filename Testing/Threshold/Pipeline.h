/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <stdio.h>
#include <iostream>
#include "Timer.h"

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/worklet/Elevation.h>
#include <dax/cont/worklet/Threshold.h>

#include <vector>

#define MAKE_STRING2(x) #x
#define MAKE_STRING1(x) MAKE_STRING2(x)
#define DEVICE_ADAPTER MAKE_STRING1(DAX_DEFAULT_DEVICE_ADAPTER)

namespace
{

class CheckValid {
public:
  CheckValid() : Valid(true) { }
  operator bool() { return this->Valid; }
  void operator()(dax::Scalar value) {
    if ((value < -1) || (value > 1)) { this->Valid = false; }
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
        break;
        }
      }

    index++;
    }
}

void PrintResults(int pipeline, double time)
{
  std::cout << "Elapsed time: " << time << " seconds." << std::endl;
  std::cout << "CSV," DEVICE_ADAPTER ","
            << pipeline << "," << time << std::endl;
}

void RunDAXPipeline(const dax::cont::UniformGrid &grid)
{
  std::cout << "Running pipeline 1: Elevation -> Threshold" << std::endl;

  dax::cont::UnstructuredGrid<dax::exec::CellHexahedron> grid2;

  dax::cont::ArrayHandle<dax::Scalar> intermediate1(grid.GetNumberOfPoints());
  dax::cont::ArrayHandle<dax::Scalar> resultHandle;

  dax::Scalar min = 0;
  dax::Scalar max = 100;

  dax::cont::worklet::Elevation(grid, grid.GetPoints(), intermediate1);

  Timer timer;
  dax::cont::worklet::Threshold(grid,grid2,min,max,intermediate1,resultHandle);
  double time = timer.elapsed();
  std::cout << "original GetNumberOfCells: " << grid.GetNumberOfCells() << std::endl;
  std::cout << "threshold GetNumberOfCells: " << grid2.GetNumberOfCells() << std::endl;

  PrintResults(1, time);
}


} // Anonymous namespace

