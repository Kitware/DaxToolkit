/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <stdio.h>
#include <iostream>
#include "ArgumentsParser.h"
#include "Timer.h"

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/worklet/CellGradient.h>
#include <dax/cont/worklet/Cosine.h>
#include <dax/cont/worklet/Elevation.h>
#include <dax/cont/worklet/Sine.h>
#include <dax/cont/worklet/Square.h>

#include <vector>

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

dax::cont::UniformGrid CreateInputStructure(dax::Id dim)
{
  dax::cont::UniformGrid grid;
  grid.SetOrigin(dax::make_Vector3(0.0, 0.0, 0.0));
  grid.SetSpacing(dax::make_Vector3(1.0, 1.0, 1.0));
  grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(dim-1, dim-1, dim-1));
  return grid;
}

void RunPipeline1(const dax::cont::UniformGrid &grid)
{
  std::cout << "Running pipeline 1: Elevation -> Gradient" << std::endl;

  dax::cont::ArrayHandle<dax::Scalar> intermediate1(grid.GetNumberOfPoints());

  std::vector<dax::Vector3> resultsBuffer(grid.GetNumberOfCells());
  dax::cont::ArrayHandle<dax::Vector3> results(resultsBuffer.begin(),
                                               resultsBuffer.end());

  Timer timer;
  dax::cont::worklet::Elevation(grid, grid.GetPoints(), intermediate1);
  dax::cont::worklet::CellGradient(grid,
                                         grid.GetPoints(),
                                         intermediate1,
                                         results);
  double time = timer.elapsed();

  PrintCheckValues(resultsBuffer.begin(), resultsBuffer.end());

  std::cout << "Elapsed time: " << time << " seconds." << std::endl;
  std::cout << "CSV,1," << time << std::endl;
}

void RunPipeline2(const dax::cont::UniformGrid &grid)
{
  std::cout << "Running pipeline 2: Elevation->Gradient->Sine->Square->Cosine"
            << std::endl;

  dax::cont::ArrayHandle<dax::Scalar> intermediate1(grid.GetNumberOfPoints());
  dax::cont::ArrayHandle<dax::Vector3> intermediate2(grid.GetNumberOfCells());
  dax::cont::ArrayHandle<dax::Vector3> intermediate3(grid.GetNumberOfCells());

  std::vector<dax::Vector3> resultsBuffer(grid.GetNumberOfCells());
  dax::cont::ArrayHandle<dax::Vector3> results(resultsBuffer.begin(),
                                               resultsBuffer.end());

  Timer timer;
  dax::cont::worklet::Elevation(grid, grid.GetPoints(), intermediate1);
  dax::cont::worklet::CellGradient(grid,
                                         grid.GetPoints(),
                                         intermediate1,
                                         intermediate2);
  intermediate1.ReleaseExecutionResources();
  dax::cont::worklet::Sine(grid, intermediate2, intermediate3);
  dax::cont::worklet::Square(grid, intermediate3, intermediate2);
  intermediate3.ReleaseExecutionResources();
  dax::cont::worklet::Cosine(grid, intermediate2, results);
  double time = timer.elapsed();

  PrintCheckValues(resultsBuffer.begin(), resultsBuffer.end());

  std::cout << "Elapsed time: " << time << " seconds." << std::endl;
  std::cout << "CSV,2," << time << std::endl;
}

void RunPipeline3(const dax::cont::UniformGrid &grid)
{
  std::cout << "Running pipeline 3: Elevation -> Sine -> Square -> Cosine"
            << std::endl;

  dax::cont::ArrayHandle<dax::Scalar> intermediate1(grid.GetNumberOfPoints());
  dax::cont::ArrayHandle<dax::Scalar> intermediate2(grid.GetNumberOfPoints());

  std::vector<dax::Scalar> resultsBuffer(grid.GetNumberOfPoints());
  dax::cont::ArrayHandle<dax::Scalar> results(resultsBuffer.begin(),
                                              resultsBuffer.end());

  Timer timer;
  dax::cont::worklet::Elevation(grid, grid.GetPoints(), intermediate1);
  dax::cont::worklet::Sine(grid, intermediate1, intermediate2);
  dax::cont::worklet::Square(grid, intermediate2, intermediate1);
  intermediate2.ReleaseExecutionResources();
  dax::cont::worklet::Cosine(grid, intermediate1, results);
  double time = timer.elapsed();

  PrintCheckValues(resultsBuffer.begin(), resultsBuffer.end());

  std::cout << "Elapsed time: " << time << " seconds." << std::endl;
  std::cout << "CSV,3," << time << std::endl;
}

} // Anonymous namespace


int main(int argc, char* argv[])
  {
  dax::testing::ArgumentsParser parser;
  if (!parser.parseArguments(argc, argv))
    {
    return 1;
    }

  //init grid vars from parser
  const dax::Id MAX_SIZE = parser.problemSize();

  dax::cont::UniformGrid grid = CreateInputStructure(MAX_SIZE);

  int pipeline = parser.pipeline();
  std::cout << "Pipeline #" << pipeline << std::endl;
  switch (pipeline)
    {
    case dax::testing::ArgumentsParser::CELL_GRADIENT:
      RunPipeline1(grid);
      break;
    case dax::testing::ArgumentsParser::CELL_GRADIENT_SINE_SQUARE_COS:
      RunPipeline2(grid);
      break;
    case dax::testing::ArgumentsParser::SINE_SQUARE_COS:
      RunPipeline3(grid);
      break;
    default:
      std::cout << "No pipeline selected." << std::endl;
      break;
    }

  return 0;
}
