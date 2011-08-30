/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include <dax/cont/DataArrayStructuredPoints.h>

//-----------------------------------------------------------------------------
dax::cont::DataArrayStructuredPoints::DataArrayStructuredPoints()
{
  this->HeavyData.Origin.x = this->HeavyData.Origin.y = this->HeavyData.Origin.z = 0;
  this->HeavyData.Spacing.x = this->HeavyData.Spacing.y = this->HeavyData.Spacing.z = 1;
  this->HeavyData.ExtentMin.x = this->HeavyData.ExtentMin.y =
    this->HeavyData.ExtentMin.z = -1;
  this->HeavyData.ExtentMax.x = this->HeavyData.ExtentMax.y =
    this->HeavyData.ExtentMax.z = 0;
}

//-----------------------------------------------------------------------------
dax::cont::DataArrayStructuredPoints::~DataArrayStructuredPoints()
{

}
