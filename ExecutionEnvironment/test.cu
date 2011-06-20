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
  //cell_attribute.Set(work, scalar);
}
  

__global__ void Test()
{
  DaxWorkMapCell work;
  DaxCell cell(work);
}

int main()
{
  Test<<<1, 10>>>();
}
