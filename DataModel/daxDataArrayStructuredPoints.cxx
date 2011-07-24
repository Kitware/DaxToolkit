/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxDataArrayStructuredPoints.h"

#include "DaxDataArray.h"
//-----------------------------------------------------------------------------
daxDataArrayStructuredPoints::daxDataArrayStructuredPoints()
{
  this->HeavyData.Origin.x = this->HeavyData.Origin.y = this->HeavyData.Origin.z = 0;
  this->HeavyData.Spacing.x = this->HeavyData.Spacing.y = this->HeavyData.Spacing.z = 1;
  this->HeavyData.ExtentMin.x = this->HeavyData.ExtentMin.y =
    this->HeavyData.ExtentMin.z = -1;
  this->HeavyData.ExtentMax.x = this->HeavyData.ExtentMax.y =
    this->HeavyData.ExtentMax.z = 0;
}

//-----------------------------------------------------------------------------
daxDataArrayStructuredPoints::~daxDataArrayStructuredPoints()
{
}

//-----------------------------------------------------------------------------
DaxDataArray daxDataArrayStructuredPoints::Upload(bool copy_heavy_data/*=false*/)
{
  return copy_heavy_data?
    DaxDataArray::CreateAndCopy(
      DaxDataArray::STRUCTURED_POINTS,
      DaxDataArray::VECTOR3,
      sizeof(DaxStructuredPointsMetaData),
      &this->HeavyData) :
    DaxDataArray::Create(
      DaxDataArray::STRUCTURED_POINTS,
      DaxDataArray::VECTOR3,
      sizeof(DaxStructuredPointsMetaData));
}

//-----------------------------------------------------------------------------
bool daxDataArrayStructuredPoints::Download(const DaxDataArray& array)
{
  assert (array.SizeInBytes == sizeof(DaxStructuredPointsMetaData));
  return array.CopyTo(&this->HeavyData, sizeof(DaxStructuredPointsMetaData));
}
