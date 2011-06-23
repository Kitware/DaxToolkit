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

DAX_WORKLET void CellGradient(DAX_IN DaxWorkMapCell& work,
  DAX_IN DaxFieldCoordinates points,
  DAX_IN DaxFieldPoint point_attribute,
  DAX_OUT DaxFieldCell& cell_attribute)
{
  DaxScalar scalar = work.GetItem();
  DaxVector3 vec = make_DaxVector3(scalar, scalar, scalar);
  cell_attribute.Set(work, vec);
}

__global__ void Execute(DaxDataObject input_do, DaxDataObject output_p2c,
  DaxArray output_cg)
{
  DaxWorkMapCell work(input_do.CellArray);
  DaxFieldPoint in_point_scalars(input_do.PointData);
  DaxFieldCell out_cell_scalars(output_p2c.CellData);
  PointDataToCellData(work, in_point_scalars, out_cell_scalars);

  DaxFieldCoordinates in_points(input_do.PointCoordinates);
  DaxFieldCell out_cell_scalars_cg(output_cg);
  CellGradient(work, in_points, in_point_scalars, out_cell_scalars_cg);
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

  DaxArrayIrregular cell_scalars_p2c;
  cell_scalars_p2c.SetNumberOfTuples(CELL_EXTENT*CELL_EXTENT*CELL_EXTENT);
  cell_scalars_p2c.SetNumberOfComponents(1);
  cell_scalars_p2c.Allocate();
  for (int cc=0; cc < CELL_EXTENT*CELL_EXTENT*CELL_EXTENT; cc++)
    {
    cell_scalars_p2c.SetValue(cc, 0, -1);
    }

  DaxArrayIrregular cell_scalars_cg;
  cell_scalars_cg.SetNumberOfTuples(CELL_EXTENT*CELL_EXTENT*CELL_EXTENT);
  cell_scalars_cg.SetNumberOfComponents(3);
  cell_scalars_cg.Allocate();
  for (int cc=0; cc < CELL_EXTENT*CELL_EXTENT*CELL_EXTENT; cc++)
    {
    for (int kk=0; kk < 3; kk++)
      {
      cell_scalars_cg.SetValue(cc, kk, -1);
      }
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

  DaxDataObject output_p2c;
  output_p2c.CellData = cell_scalars_p2c;

  DaxDataObject output_cg;
  output_cg.CellData = cell_scalars_cg;

  DaxDataObjectDevice d_input; d_input.CopyFrom(input);
  DaxDataObjectDevice d_output_p2c; d_output_p2c.Allocate(output_p2c);
  DaxDataObjectDevice d_output_cg; d_output_cg.Allocate(output_cg);

  Execute<<<CELL_EXTENT, CELL_EXTENT*CELL_EXTENT>>>(d_input,
    d_output_p2c, d_output_cg.CellData);

  output_p2c.CopyFrom(d_output_p2c);
  for (int cc=0; cc < CELL_EXTENT*CELL_EXTENT*CELL_EXTENT; cc++)
    {
    cout << cell_scalars_p2c.GetValue(cc, 0) << endl;
    }

  output_cg.CopyFrom(d_output_cg);
  for (int cc=0; cc < CELL_EXTENT*CELL_EXTENT*CELL_EXTENT; cc++)
    {
    for (int kk=0; kk < 3; kk++)
      {
      cout << cell_scalars_cg.GetValue(cc, kk) << ", ";
      }
    cout << endl;
    }

  d_input.FreeMemory();
  input.FreeMemory();

  d_output_p2c.FreeMemory();
  output_p2c.FreeMemory();

  d_output_cg.FreeMemory();
  output_cg.FreeMemory();
  return 0;
}
