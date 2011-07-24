/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxImageData.h"
#include "daxDataArrayStructuredPoints.h"
#include "daxDataArrayStructuredConnectivity.h"

//-----------------------------------------------------------------------------
daxImageData::daxImageData()
{
  this->Extent[0] = this->Extent[2] = this->Extent[4] = 0;
  this->Extent[1] = this->Extent[3] = this->Extent[5] = -1;
  this->Origin.x = this->Origin.y = this->Origin.z = 0;
  this->Spacing.x = this->Spacing.y = this->Spacing.z = 1;
  this->PointCoordinates = daxDataArrayStructuredPointsPtr(
    new daxDataArrayStructuredPoints());
  this->CellArray = daxDataArrayStructuredConnectivityPtr(
    new daxDataArrayStructuredConnectivity());
}

//-----------------------------------------------------------------------------
daxImageData::~daxImageData()
{
}

//-----------------------------------------------------------------------------
daxDataArrayPtr daxImageData::GetPointCoordinates() const
{
  this->PointCoordinates->SetOrigin(this->Origin);
  this->PointCoordinates->SetSpacing(this->Spacing);
  this->PointCoordinates->SetExtents(
    this->Extent[0], this->Extent[1], this->Extent[2],
    this->Extent[3], this->Extent[4], this->Extent[5]);
  return this->PointCoordinates; 
}

//-----------------------------------------------------------------------------
daxDataArrayPtr daxImageData::GetCellArray() const
{
  this->CellArray->SetOrigin(this->Origin);
  this->CellArray->SetSpacing(this->Spacing);
  this->CellArray->SetExtents(
    this->Extent[0], this->Extent[1], this->Extent[2],
    this->Extent[3], this->Extent[4], this->Extent[5]);
  return this->CellArray; 
}
