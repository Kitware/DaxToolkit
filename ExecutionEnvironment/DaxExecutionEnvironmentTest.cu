/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include "DaxExecutionEnvironment.h"

#include "PointDataToCellData.worklet"
#include "CellGradient.worklet"
#include "CellAverage.worklet"

__global__ void Execute()
{
//  DaxWorkMapCell work(input_do.CellArray);
//  DaxFieldPoint in_point_scalars(input_do.PointData);
//  DaxFieldCell out_cell_scalars(output_p2c.CellData);
//
//  PointDataToCellData(work, in_point_scalars, out_cell_scalars);
//  //CellAverage(work, in_point_scalars, out_cell_scalars);
//
//  DaxFieldCoordinates in_points(input_do.PointCoordinates);
//  DaxFieldCell out_cell_scalars_cg(output_cg);
//  CellGradient(work, in_points, in_point_scalars, out_cell_scalars_cg);
}

#include <iostream>
using namespace std;
#define POINT_EXTENT 128
#define CELL_EXTENT 127
int main()
{
  return 0;
}
