/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxImageData_h
#define __daxImageData_h

#include "daxDataSet.h"
#include "daxTypes.h"

/// daxImageData is the abstract superclass for data array object containing
/// numeric data.
class daxImageData : public daxDataSet
{
public:
  daxImageData();
  virtual ~daxImageData();
  daxTypeMacro(daxImageData, daxDataSet);

  /// Get/Set the image extents.
  void SetExtent(int minx, int maxx, int miny, int maxy, int minz, int maxz)
    {
    this->Extent[0] = minx; this->Extent[1] = maxx;
    this->Extent[2] = miny; this->Extent[3] = maxy;
    this->Extent[4] = minz; this->Extent[5] = maxz;
    }
  const int* GetExtent() const
    { return this->Extent; }

  void SetOrigin(const DaxVector3& origin)
    { this->Origin = origin; }
  void SetOrigin(double x, double y, double z)
    {
    this->Origin.x = x;
    this->Origin.y = y;
    this->Origin.z = z;
    }

  const DaxVector3& GetOrigin() const
    { return this->Origin; }

  void SetSpacing(double x, double y, double z)
    {
    this->Spacing.x = x;
    this->Spacing.y = y;
    this->Spacing.z = z;
    }
  void SetSpacing(const DaxVector3& spacing)
    { this->Spacing = spacing; }
  const DaxVector3& GetSpacing() const
    { return this->Spacing; }

  /// Get number of points;
  int GetNumberOfPoints() const
    {
    return (this->Extent[1] - this->Extent[0] + 1) *
      (this->Extent[3] - this->Extent[2] + 1) *
       (this->Extent[5] - this->Extent[4] + 1);
    }

  /// Get number of cells;
  int GetNumberOfCells() const
    {
    return (this->Extent[1] - this->Extent[0]) *
      (this->Extent[3] - this->Extent[2]) *
       (this->Extent[5] - this->Extent[4]);
    }

  virtual daxDataArrayPtr GetPointCoordinates() const;
  virtual daxDataArrayPtr GetCellArray() const;

protected:
  int Extent[6];
  DaxVector3 Origin;
  DaxVector3 Spacing;
private:
  daxDisableCopyMacro(daxImageData)
};

/// declares daxImageDataPtr
daxDefinePtrMacro(daxImageData)

#endif
