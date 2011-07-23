/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxDataArrayStructuredPoints_h
#define __daxDataArrayStructuredPoints_h

#include "daxDataArray.h"
#include "daxTypes.h"

/// daxDataArrayStructuredPoints is the abstract superclass for data array object containing
/// numeric data.
class daxDataArrayStructuredPoints : public daxDataArray
{
public:
  daxDataArrayStructuredPoints();
  virtual ~daxDataArrayStructuredPoints();
  daxTypeMacro(daxDataArrayStructuredPoints, daxDataArray);

  void SetOrigin(const DaxVector3& origin)
    { this->HeavyData.Origin = origin; }
  const DaxVector3& GetOrigin() const
    { return this->HeavyData.Origin; }

  void SetSpacing(const DaxVector3& spacing)
    { this->HeavyData.Spacing = spacing; }
  const DaxVector3& GetSpacing() const
    { return this->HeavyData.Spacing; }

  void SetExtents(int minx, int maxx, int miny, int maxy, int minz, int maxz)
    {
    this->HeavyData.ExtentMin.x = minx;
    this->HeavyData.ExtentMin.y = miny;
    this->HeavyData.ExtentMin.z = minz;
    this->HeavyData.ExtentMax.x = maxx;
    this->HeavyData.ExtentMax.y = maxy;
    this->HeavyData.ExtentMax.z = maxz;
    }

  /// Called to convert the array to a DaxDataArray which can be passed the
  /// Execution environment.
  virtual bool Convert(DaxDataArray* array);

protected:
  DaxStructuredPointsMetaData HeavyData;

private:
  daxDisableCopyMacro(daxDataArrayStructuredPoints)
};

daxDefinePtrMacro(daxDataArrayStructuredPoints)
#endif
