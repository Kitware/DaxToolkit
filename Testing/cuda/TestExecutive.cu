/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <stdio.h>
#include <iostream>

#include <dax/cont/StructuredGrid.h>
#include <dax/cuda/cont/Executive.h>
#include <dax/cuda/cont/MapCellModule.h>
#include <dax/cuda/cont/MapFieldModule.h>
#include <dax/cuda/cont/Filter.h>
#include <dax/cont/Array.h>
#include "Worklets.h"


static dax::cont::StructuredGrid CreateInputStructure(dax::Id dim)
{
  dax::cont::StructuredGrid grid(
    dax::make_Vector3(0.0, 0.0, 0.0),
    dax::make_Vector3(1.0, 1.0, 1.0),
    dax::make_Id3(0, 0, 0),
    dax::make_Id3(dim-1, dim-1, dim-1) );

  return grid;
}

static void PrintCheckValues(const dax::internal::DataArray<dax::Vector3> &array)
{
  std::cout << "PrintCheckValues" << std::endl;
  for (dax::Id index = 0; index < array.GetNumberOfEntries(); index++)
    {
    dax::Vector3 value = array.GetValue(index);
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

template<typename T>
void ExecutePipeline(T& data)
{
  using namespace dax::cont;


  dax::cont::Array<modules::CellGradient::OutputType> gradientResults;

  dax::cuda::cont::Model<T> model(data);

  Filter<modules::Elevation> elev(model,PointField());
  Filter<modules::CellGradient> gradient(model,elev);

  Pull(gradient,gradientResults);
  PrintCheckValues(gradientResults);

  data.addCellField(&gradientResults);
}

int main(int argc, char* argv[])
{
  dax::cont::StructuredGrid grid = CreateInputStructure(32);
  ExecutePipeline(grid);
  return 0;
}
