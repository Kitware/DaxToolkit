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

#include <dax/cuda/cont/worklet/Cosine.h>
#include <dax/cuda/cont/worklet/Elevation.h>
#include <dax/cuda/cont/worklet/Sine.h>
#include <dax/cuda/cont/worklet/Square.h>

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
  dax::cuda::cont::worklet::Elevation(grid, intermediate1);
  dax::cuda::cont::worklet::Sine(grid, intermediate1, intermediate2);
  dax::cuda::cont::worklet::Square(grid, intermediate2, intermediate1);
  intermediate2.ReleaseExecutionResources();
  dax::cuda::cont::worklet::Cosine(grid, intermediate1, results);
  double time = timer.elapsed();

  PrintCheckValues(resultsBuffer.begin(), resultsBuffer.end());

  std::cout << "Elapsed time: " << time << " seconds." << std::endl;
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

  RunPipeline3(grid);

  return 0;
}

#if 0

void addArrays(dax::cont::StructuredGrid &grid)
{
  dax::cont::ArrayPtr<dax::Scalar> testPData(
        new dax::cont::Array<dax::Scalar>());
  testPData->resize(grid.numPoints(),1);
  grid.fieldsPoint().addArray("pointArray",testPData);

  dax::cont::ArrayPtr<dax::Scalar> testCData(
        new dax::cont::Array<dax::Scalar>());
  testCData->resize(grid.numCells(),1);

  grid.fieldsCell().addArray("cellArray",testCData);
}

void ElevationWorklet(dax::cont::StructuredGrid& grid, const std::string &name)
{
  dax::cont::worklets::Elevation(grid,
    grid.points(),
    dax::cont::FieldHandlePoint<dax::Scalar>(name));
}

void CellGradientWorklet(dax::cont::StructuredGrid& grid,
                         const std::string &inputName,
                         const std::string &resultName)
{
  dax::cont::worklets::CellGradient(grid,
    grid.points(),
    grid.fieldsPoint().scalar(inputName),
    dax::cont::FieldHandleCell<dax::Vector3>(resultName));
}

template<typename T>
void PointSineSquareCosWorklet(
    T t,
    dax::cont::StructuredGrid& grid,
    const std::string &inputName,
    const std::string &resultName)
{
  dax::cont::worklets::Sine(grid,
                            grid.fieldsPoint().get(T(),inputName),
                            dax::cont::FieldHandlePoint<T>(resultName));
  dax::cont::worklets::Square(grid,
                              grid.fieldsPoint().get(T(),resultName),
                              dax::cont::FieldHandlePoint<T>(grid,resultName));
  dax::cont::worklets::Cosine(grid,
                              grid.fieldsPoint().get(T(),resultName),
                              dax::cont::FieldHandlePoint<T>(grid,resultName));  
}

template<typename T>
void CellSineSquareCosWorklet(
    T t,
    dax::cont::StructuredGrid& grid,
    const std::string &inputName,
    const std::string &resultName)
{
  dax::cont::worklets::Sine(grid,
                            grid.fieldsCell().get(T(),inputName),
                            dax::cont::FieldHandleCell<T>(resultName));

  dax::cont::worklets::Square(grid,
                              grid.fieldsCell().get(T(),resultName),
                              dax::cont::FieldHandleCell<T>(grid,resultName));

  dax::cont::worklets::Cosine(grid,
                              grid.fieldsCell().get(T(),resultName),
                              dax::cont::FieldHandleCell<T>(grid,resultName));
}

void Pipeline1(dax::cont::StructuredGrid &grid, double& execute_time)
  {
  boost::timer timer;

  timer.restart();
  ElevationWorklet(grid, "Elevation");
  CellGradientWorklet(grid, "Elevation", "Gradient");
  execute_time = timer.elapsed();

  dax::cont::ArrayPtr<dax::Vector3> result(
        dax::cont::retrieve(grid.fieldsCell().vector3("Gradient")));
  PrintCheckValues(result);
  }

void Pipeline2(dax::cont::StructuredGrid &grid, double& execute_time)
  {
  boost::timer timer;

  //let pipeline 1 time itself
  Pipeline1(grid,execute_time);

  timer.restart();
  CellSineSquareCosWorklet(dax::Vector3(), grid,"Gradient","Result");
  execute_time += timer.elapsed();

  PrintCheckValues(dax::cont::retrieve(grid.fieldsCell().vector3("Result")));
  }

void Pipeline3(dax::cont::StructuredGrid &grid, double& execute_time)
  {
  boost::timer timer;
  timer.restart();

  ElevationWorklet(grid,"Elevation");
  PointSineSquareCosWorklet(dax::Scalar(),grid,"Elevation","Result");
  execute_time = timer.elapsed();


  PrintCheckValues(dax::cont::retrieve(grid.fieldsPoint().scalar("Result")));
  }
}

int main(int argc, char* argv[])
  {

  dax::testing::ArgumentsParser parser;
  if (!parser.parseArguments(argc, argv))
    {
    return 1;
    }

  //init grid vars from parser
  const dax::Id MAX_SIZE = parser.problemSize();

  //init timer vars
  boost::timer timer;
  double execute_time = 0;
  double init_time = 0;

  //create the grid
  timer.restart();
  dax::cont::StructuredGrid grid;
  CreateInputStructure(MAX_SIZE,grid);
  init_time = timer.elapsed();


  int pipeline = parser.pipeline();
  std::cout << "Pipeline #"<<pipeline<< std::endl;
  switch (pipeline)
    {
  case dax::testing::ArgumentsParser::CELL_GRADIENT:
    Pipeline1(grid,execute_time);
    break;
  case dax::testing::ArgumentsParser::CELL_GRADIENT_SINE_SQUARE_COS:
    Pipeline2(grid,execute_time);
    break;
  case dax::testing::ArgumentsParser::SINE_SQUARE_COS:
    Pipeline3(grid,execute_time);
    break;
  default:
    addArrays(grid);
    break;
    }

  std::cout << std::endl << std::endl
            << "Summary: -- " << MAX_SIZE << "^3 Dataset" << std::endl;
  std::cout << "Initialize: " << init_time << std::endl
            << "Execute: " << execute_time << std::endl;

  return 0;
  }

#endif
