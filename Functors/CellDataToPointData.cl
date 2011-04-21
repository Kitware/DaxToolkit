/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// Functor that converts cell data to point data.
void CellDataToPointData(const daxWork* work,
  const daxArray* __connections__ cell_links,
  const daxArray* __dep__(__connections__) in_cell_array,
  daxArray* __dep__(__positions__) out_point_array)
{
  daxFloat3 sum;
  daxWork new_work;
  new_work.ElementID = 0;
  sum = daxGetArrayValue3(&new_work, in_cell_array);
  daxSetArrayValue3(work, out_point_array, sum);

  //daxConnectedComponent cells_containing_point;
  //daxGetConnectedComponent(work, cell_links, &cells_containing_point);

  //daxIdType num_cells = daxGetNumberOfElements(&cells_containing_point);
  //daxFloat weight = 1.0 / as_float(num_cells);
  //daxWork cell_work;
  //daxFloat3 sum;
  //sum.x = sum.y = sum.z = 0;
  //for (daxIdType cc=0; cc < num_cells; cc++)
  //  {
  //  daxGetWorkForElement(&cells_containing_point, cc, &cell_work);
  //  sum = daxGetArrayValue3(&cell_work, in_cell_array);
  //  break;
  //  //daxFloat3 value = daxGetArrayValue3(&cell_work, in_cell_array);
  //  //sum += (value * weight);
  //  }
  //daxSetArrayValue3(work, out_point_array, sum);
  //daxSetArrayValue(work, out_point_array, sum.x);
}
