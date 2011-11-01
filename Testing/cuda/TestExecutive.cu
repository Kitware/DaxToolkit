/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <stdio.h>
#include <iostream>

#include <dax/internal/GridStructures.h>
#include "Modules.h"
#include "Executive.h"

static dax::internal::StructureUniformGrid CreateInputStructure(dax::Id dim)
{
  dax::internal::StructureUniformGrid grid;
  grid.Origin = dax::make_Vector3(0.0, 0.0, 0.0);
  grid.Spacing = dax::make_Vector3(1.0, 1.0, 1.0);
  grid.Extent.Min = dax::make_Id3(0, 0, 0);
  grid.Extent.Max = dax::make_Id3(dim-1, dim-1, dim-1);
  return grid;
}


static void PrintCheckValues(std::vector<dax::Vector3> &data)
{
  std::cout << "PrintCheckValues" << std::endl;
  dax::internal::DataArray<dax::Vector3> array;
  array.SetPointer(&data[0],data.size());

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
void ExecutePipeline(T data)
{
  std::vector<dax::modules::GradientM::OutputDataType> results;

  dax::exec::Source<T> s(data); //name: Push?
  dax::exec::Filter< dax::modules::ElevationM > filter1(s);
  dax::exec::Filter< dax::modules::GradientM > filter2(s,filter1);


  dax::exec::Sink(filter2,results); //name: Pull? Fetch?
  PrintCheckValues(results);
}


int main(int argc, char* argv[])
{
  dax::internal::StructureUniformGrid grid = CreateInputStructure(32);
  ExecutePipeline(grid);

  return 1;
}
