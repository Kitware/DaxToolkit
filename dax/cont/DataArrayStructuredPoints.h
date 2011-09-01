/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_DataArrayStructuredPoints_h
#define __dax_cont_DataArrayStructuredPoints_h

#include <dax/Types.h>

#include <dax/cont/DataArray.h>

namespace dax { namespace cont {

daxDeclareClass(DataArrayStructuredPoints);

/// DataArrayStructuredPoints is the abstract superclass for data array object containing
/// numeric data.
class DataArrayStructuredPoints : public dax::cont::DataArray
{
public:
  DataArrayStructuredPoints();
  virtual ~DataArrayStructuredPoints();
  daxTypeMacro(DataArrayStructuredPoints, dax::cont::DataArray);

  void SetOrigin(const dax::Vector3& origin)
    { this->HeavyData.Origin = origin; }
  const dax::Vector3& GetOrigin() const
    { return this->HeavyData.Origin; }

  void SetSpacing(const dax::Vector3& spacing)
    { this->HeavyData.Spacing = spacing; }
  const dax::Vector3& GetSpacing() const
    { return this->HeavyData.Spacing; }

  void SetExtents(int minx, int maxx, int miny, int maxy, int minz, int maxz)
    {
    this->HeavyData.Extent.Min.x = minx;
    this->HeavyData.Extent.Min.y = miny;
    this->HeavyData.Extent.Min.z = minz;
    this->HeavyData.Extent.Max.x = maxx;
    this->HeavyData.Extent.Max.y = maxy;
    this->HeavyData.Extent.Max.z = maxz;
    }

  const dax::StructuredPointsMetaData& GetHeavyData() const
    { return this->HeavyData; }

protected:
  dax::StructuredPointsMetaData HeavyData;

private:
  daxDisableCopyMacro(DataArrayStructuredPoints)
};

}}
#endif
