/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// Functor that computes cell-gradient based on point-scalars.
void CellGradient2(const daxWork* work,
  const daxArray* __positions__ in_positions,
  const daxArray* __and__(__connections__, __ref__(in_positions)) in_connections,
  const daxArray* __dep__(in_positions) inputArray,
  daxArray* __dep__(in_connections) outputArray)
{
  // Get the connected-components using the connections array.
  daxConnectedComponent cell;
  daxGetConnectedComponent(work, in_connections, &cell);

  float scalars[8];
  uint num_elements = daxGetNumberOfElements(&cell);
  daxWork point_work;
  for (uint cc=0; cc < num_elements; cc++)
    {
    daxGetWorkForElement(&cell, cc, &point_work);    
    scalars[cc] = daxGetArrayValue(&point_work, inputArray);
    }

  float4 parametric_cell_center = (float4)(0.5, 0.5, 0.5, 0);
  float3 gradient = daxGetCellDerivative(&cell,
    0, parametric_cell_center, scalars);

  daxSetArrayValue3(work, outputArray, gradient);

  printf("Cell Gradient: %f\n", gradient);
  //  float in_value = daxGetArrayValue(work, inputArray) * 2.0;
  //  daxSetArrayValue(work, outputArray, in_value);
}
