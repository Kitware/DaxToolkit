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

  daxFloat scalars[8];
  daxIdType num_elements = daxGetNumberOfElements(&cell);
  daxWork point_work;
  for (daxIdType cc=0; cc < num_elements; cc++)
    {
    daxGetWorkForElement(&cell, cc, &point_work);    
    daxFloat3 temp = daxGetArrayValue3(&point_work, inputArray);
    scalars[cc] = sqrt(temp.x*temp.x + temp.y*temp.y + temp.z*temp.z);
    }

  daxFloat3 parametric_cell_center = as_daxFloat3(0.5, 0.5, 0.5);
  daxFloat3 gradient = daxGetCellDerivative(&cell,
    0, parametric_cell_center, scalars);
  daxSetArrayValue3(work, outputArray, gradient);

  printf("Cell Gradient: %f, %f, %f \n", gradient.x, gradient.y, gradient.z);
}
