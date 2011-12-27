/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include <dax/cont/DataArrayStructuredPoints.h>

//-----------------------------------------------------------------------------
dax::cont::DataArrayStructuredPoints::DataArrayStructuredPoints()
{
  this->HeavyData.Origin[0] = this->HeavyData.Origin[1] = this->HeavyData.Origin[2] = 0;
  this->HeavyData.Spacing[0] = this->HeavyData.Spacing[1] = this->HeavyData.Spacing[2] = 1;
  this->HeavyData.Extent.Min[0] = this->HeavyData.Extent.Min[1] =
    this->HeavyData.Extent.Min[2] = -1;
  this->HeavyData.Extent.Max[0] = this->HeavyData.Extent.Max[1] =
    this->HeavyData.Extent.Max[2] = 0;
}

//-----------------------------------------------------------------------------
dax::cont::DataArrayStructuredPoints::~DataArrayStructuredPoints()
{

}
