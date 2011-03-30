/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// Functor that computes cell-scalars based on point-scalars.
void CellAverage(const daxWork* work,
  const daxArray* __positions__ in_positions,
  const daxArray* __and__(__connections__, __ref__(in_positions)) in_connections,
  const daxArray* __dep__(in_positions) inputArray,
  daxArray* __dep__(in_connections) outputArray)
{
//  // Get the connected-components using the connections array.
//  daxConnectedComponent cell;
//  daxGetConnectedComponent(work, in_connections, &cell);
//
//  float sum_value = 0.0;
//  for (int cc=0; cc < daxGetNumberOfElements(&cell); cc++)
//    {
//    // Generate a "work" for the point of interest.
//    daxWork point_work;
//    daxGetWorkForElement(&cell, cc, &point_work);
//
//    sum_value += daxGetArrayValue(point_work, inputArray);
//    }
//  sum_value /= daxGetNumberOfElements(&cell);
//  daxSetArrayValue(work, outputArray, sum_value);
  float in_value = daxGetArrayValue(work, inputArray) * 2.0;
  daxSetArrayValue(work, outputArray, in_value);
}
