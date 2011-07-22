/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxImageData.h"
#include "daxDataArrayStructuredPoints.h"

//-----------------------------------------------------------------------------
daxImageData::daxImageData()
{
  this->Extent[0] = this->Extent[2] = this->Extent[4] = 0;
  this->Extent[1] = this->Extent[3] = this->Extent[5] = -1;
  this->Origin.x = this->Origin.y = this->Origin.z = 0;
  this->Spacing.x = this->Spacing.y = this->Spacing.z = 1;
}

//-----------------------------------------------------------------------------
daxImageData::~daxImageData()
{
}

//-----------------------------------------------------------------------------
daxDataArrayPtr daxImageData::GetPointCoordinates() const
{
  daxDataArrayStructuredPointsPtr point_coordinates(new
    daxDataArrayStructuredPoints);
  point_coordinates->SetOrigin(this->Origin);
  point_coordinates->SetSpacing(this->Spacing);
  point_coordinates->SetExtents(
    this->Extent[0], this->Extent[1], this->Extent[2],
    this->Extent[3], this->Extent[4], this->Extent[5]);
  return point_coordinates; 
}

//-----------------------------------------------------------------------------
daxDataArrayPtr daxImageData::GetCellArray() const
{
  return daxDataArrayPtr();
}
