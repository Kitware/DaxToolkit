/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include <dax/cont/ImageData.h>

#include <dax/cont/DataArrayStructuredConnectivity.h>
#include <dax/cont/DataArrayStructuredPoints.h>

//-----------------------------------------------------------------------------
dax::cont::ImageData::ImageData()
{
  this->Extent[0] = this->Extent[2] = this->Extent[4] = 0;
  this->Extent[1] = this->Extent[3] = this->Extent[5] = -1;
  this->Origin.x = this->Origin.y = this->Origin.z = 0;
  this->Spacing.x = this->Spacing.y = this->Spacing.z = 1;
  this->PointCoordinates = dax::cont::DataArrayStructuredPointsPtr(
    new dax::cont::DataArrayStructuredPoints());
  this->CellArray = dax::cont::DataArrayStructuredConnectivityPtr(
    new dax::cont::DataArrayStructuredConnectivity());
}

//-----------------------------------------------------------------------------
dax::cont::ImageData::~ImageData()
{
}

//-----------------------------------------------------------------------------
dax::cont::DataArrayPtr dax::cont::ImageData::GetPointCoordinates() const
{
  this->PointCoordinates->SetOrigin(this->Origin);
  this->PointCoordinates->SetSpacing(this->Spacing);
  this->PointCoordinates->SetExtents(
    this->Extent[0], this->Extent[1], this->Extent[2],
    this->Extent[3], this->Extent[4], this->Extent[5]);
  return this->PointCoordinates; 
}

//-----------------------------------------------------------------------------
dax::cont::DataArrayPtr dax::cont::ImageData::GetCellArray() const
{
  this->CellArray->SetOrigin(this->Origin);
  this->CellArray->SetSpacing(this->Spacing);
  this->CellArray->SetExtents(
    this->Extent[0], this->Extent[1], this->Extent[2],
    this->Extent[3], this->Extent[4], this->Extent[5]);
  return this->CellArray; 
}
