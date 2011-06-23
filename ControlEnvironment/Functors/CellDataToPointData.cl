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
  daxConnectedComponent cells_containing_point;
  daxGetConnectedComponent(work, cell_links, &cells_containing_point);
  daxIdType num_cells = daxGetNumberOfElements(&cells_containing_point);

  daxWork cell_work;
  daxFloat3 sum = as_daxFloat3(0, 0, 0);
  for (daxIdType cc=0; cc < num_cells; cc++)
    {
    daxGetWorkForElement(&cells_containing_point, cc, &cell_work);
    daxFloat3 value = daxGetArrayValue3(&cell_work, in_cell_array);
    value /= as_daxFloat3(num_cells, num_cells, num_cells);
    sum += value;
    }
  daxSetArrayValue3(work, out_point_array, sum);
}
