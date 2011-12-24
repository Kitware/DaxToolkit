/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <stdio.h>
#include <iostream>

#include <dax/cont/StructuredGrid.h>
#include <dax/cont/Array.h>

//has to be above dax/cont/Worklets
#include <dax/cuda/cont/Worklets.h>

//should we have a commoon all header for control?
#include <dax/cont/Worklets.h>
#include <dax/cont/FieldHandles.h>

namespace
{
void PrintCheckValues(const dax::cont::ArrayPtr<dax::Vector3> &array)
{
  for (dax::Id index = 0; index < array->size(); index++)
    {
    dax::Vector3 value = (*array)[index];
    if (index < 20)
      {
      std::cout << index << " : " << value.x << ", " << value.y << ", " << value.z
           << std::endl;
      }
    if (   (value.x < -1) || (value .x > 1)
        || (value.y < -1) || (value .y > 1)
        || (value.z < -1) || (value .z > 1) )
      {
      std::cout << index << " : " << value.x << ", " << value.y << ", " << value.z
           << std::endl;
      break;
      }
    }
}

void CreateInputStructure(dax::Id dim, dax::cont::StructuredGrid &grid )
{
  grid.Origin = dax::make_Vector3(0.0, 0.0, 0.0);
  grid.Spacing = dax::make_Vector3(1.0, 1.0, 1.0),
      grid.Extent.Min = dax::make_Id3(0, 0, 0),
      grid.Extent.Max = dax::make_Id3(dim-1, dim-1, dim-1);
  grid.computePointLocations();
}

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

void dumpArray(dax::cont::StructuredGrid& grid, const std::string &name)
{
  //TODO remove the ability to get raw array from public control API)
  dax::cont::ArrayPtr<dax::Vector3> array = dax::cont::retrieve(grid.fieldsCell().vector3(name));
  PrintCheckValues(array);
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

void Pipeline1(dax::cont::StructuredGrid &grid)
  {
  ElevationWorklet(grid, "Elevation");
  CellGradientWorklet(grid, "Elevation", "Gradient");
  }

void Pipeline2(dax::cont::StructuredGrid &grid)
  {
  Pipeline1(grid);
  CellSineSquareCosWorklet(dax::Vector3(), grid,"Gradient","Result");
  }

void Pipeline3(dax::cont::StructuredGrid &grid)
  {
  ElevationWorklet(grid,"Elevation");
  PointSineSquareCosWorklet(dax::Scalar(),grid,"Elevation","Result");
  }
}

int main(int argc, char* argv[])
  {

  dax::cont::StructuredGrid grid;
  CreateInputStructure(32,grid);

  int pipeline = 3;
  std::cout << "Pipeline #"<<pipeline<< std::endl;
  switch (pipeline)
    {
  case 1:
    Pipeline1(grid);
    break;
  case 2:
    Pipeline2(grid);
    break;
  case 3:
    Pipeline3(grid);
    break;
  default:
    addArrays(grid);
    dumpArray(grid,"cellArray");
    }
  return 0;
  }
