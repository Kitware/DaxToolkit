/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*=============================================================================
  Functor doing what vtkElevationFilter does.
=============================================================================*/
// Elevation functor that requires that positions is the input array and generates
// point-scalars.
void Elevation(const daxWork* work,
  const daxArray* __positions__ positions,
  daxArray* __dep__(positions) output)
{
  float3 in_value = daxGetArrayValue3(work, positions);
  float elevation_scalar = length(in_value);
  daxSetArrayValue(work, output, elevation_scalar);
}
