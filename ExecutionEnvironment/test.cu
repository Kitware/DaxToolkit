/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include "DaxExecutionEnvironment.h"
#include "DaxCell.cu"

DAX_WORKLET void PointDataToCellData(DAX_IN DaxWorkMapCell& work,
  DAX_IN DaxFieldPoint& point_attribute,
  DAX_OUT DaxFieldCell& cell_attribute)
{
  DaxVector3 center = make_DaxVector3(0.5, 0.5, 0.5);
  DaxCell cell(work);
  DaxScalar scalar = cell.Interpolate(center, point_attribute, 0);
  cell_attribute.Set(work, scalar);
}

__global__ void Execute(DaxDataObject input_do, DaxDataObject output_do)
{
  DaxWorkMapCell work(input_do.CellArray);
  DaxFieldPoint in_point_scalars(input_do.PointData);
  DaxFieldCell out_cell_scalars(output_do.CellData);
  PointDataToCellData(work, in_point_scalars, out_cell_scalars);
}

#include <iostream>
using namespace std;
#define POINT_EXTENT 4
#define CELL_EXTENT 3
int main()
{
  DaxArrayIrregular point_scalars;
  point_scalars.SetNumberOfTuples(POINT_EXTENT*POINT_EXTENT*POINT_EXTENT);
  point_scalars.SetNumberOfComponents(1);
  point_scalars.Allocate();
  int cc=0;
  for (int z=0; z < POINT_EXTENT; z++)
    {
    for (int y=0; y < POINT_EXTENT; y++)
      {
      for (int x=0; x < POINT_EXTENT; x++)
        {
        point_scalars.SetValue(cc, 0, cc);
        cc++;
        }
      }
    }

  DaxArrayIrregular cell_scalars;
  cell_scalars.SetNumberOfTuples(CELL_EXTENT*CELL_EXTENT*CELL_EXTENT);
  cell_scalars.SetNumberOfComponents(1);
  cell_scalars.Allocate();
  for (int cc=0; cc < CELL_EXTENT*CELL_EXTENT*CELL_EXTENT; cc++)
    {
    cell_scalars.SetValue(cc, 0, -1);
    }

  DaxArrayStructuredPoints point_coordinates;
  point_coordinates.SetExtent(0, POINT_EXTENT-1, 0, POINT_EXTENT-1, 0,
    POINT_EXTENT-1);
  point_coordinates.SetSpacing(1, 1, 1);
  point_coordinates.SetOrigin(0, 0, 0);
  point_coordinates.Allocate();

  DaxArrayStructuredConnectivity cell_array;
  cell_array.SetExtent(0, POINT_EXTENT-1, 0, POINT_EXTENT-1, 0,
    POINT_EXTENT-1);
  cell_array.SetSpacing(1, 1, 1);
  cell_array.SetOrigin(0, 0, 0);
  cell_array.Allocate();

  DaxDataObject input;
  input.PointData = point_scalars;
  input.PointCoordinates = point_coordinates;
  input.CellArray = cell_array;

  DaxDataObject output;
  output.CellData = cell_scalars;

  DaxDataObjectDevice d_input; d_input.CopyFrom(input);
  DaxDataObjectDevice d_output; d_output.Allocate(output);

  Execute<<<CELL_EXTENT, CELL_EXTENT*CELL_EXTENT>>>(d_input, d_output);

  output.CopyFrom(d_output);
  for (int cc=0; cc < CELL_EXTENT*CELL_EXTENT*CELL_EXTENT; cc++)
    {
    cout << cell_scalars.GetValue(cc, 0) << endl;
    }

  d_input.FreeMemory();
  d_output.FreeMemory();
  input.FreeMemory();
  output.FreeMemory();
  return 0;
}
