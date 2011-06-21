/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/// This file defines the DataObject that can be used on the host as well as
/// OnDevice.
#ifndef __DaxDataObject_h
#define __DaxDataObject_h

#include "DaxCommon.h"
#include "DaxArray.cu"

class DaxDataObject
{
public:
  DaxArray PointData;
  DaxArray CellData;
  DaxArray PointCoordinates;
  DaxArray CellArray;

  __host__ void FreeMemory()
    {
    this->PointData.FreeMemory();
    this->CellData.FreeMemory();
    this->PointCoordinates.FreeMemory();
    this->CellArray.FreeMemory();
    }

  __host__ void CopyFrom(const DaxDataObject& source)
    {
    this->PointData.CopyFrom(source.PointData);
    this->CellData.CopyFrom(source.CellData);
    this->PointCoordinates.CopyFrom(source.PointCoordinates);
    this->CellArray.CopyFrom(source.CellArray);
    }

  __host__ void Allocate(const DaxDataObject& source)
    {
    this->PointData.Allocate(source.PointData);
    this->CellData.Allocate(source.CellData);
    this->PointCoordinates.Allocate(source.PointCoordinates);
    this->CellArray.Allocate(source.CellArray);
    }
};

class DaxDataObjectDevice : public DaxDataObject
{
public:
  __host__ DaxDataObjectDevice()
    {
    this->PointData.OnDevice = true;
    this->CellData.OnDevice = true;
    this->PointCoordinates.OnDevice = true;
    this->CellArray.OnDevice = true;
    }
};

#endif
