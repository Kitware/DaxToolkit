/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxImageData_h
#define __daxImageData_h

#include "daxDataSet.h"

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

  /// Get/Set the origin.
  void SetOrigin(double x, double y, double z)
    { this->Origin[0] = x; this->Origin[1] = y; this->Origin[2] = z; }
  const double* GetOrigin() const
    { return this->Origin; }

  /// Get/Set spacing.
  void SetSpacing(double x, double y, double z)
    { this->Spacing[0] = x; this->Spacing[1] = y; this->Spacing[2] = z; }
  const double* GetSpacing() const
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


protected:
  int Extent[6];
  double Spacing[3];
  double Origin[3];

private:
  daxDisableCopyMacro(daxImageData)
};

/// declares daxImageDataPtr
daxDefinePtrMacro(daxImageData)

#endif
