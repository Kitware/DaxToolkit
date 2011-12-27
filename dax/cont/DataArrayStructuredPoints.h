/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_DataArrayStructuredPoints_h
#define __dax_cont_DataArrayStructuredPoints_h

#include <dax/Types.h>

#include<dax/internal/GridStructures.h>

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
    this->HeavyData.Extent.Min[0] = minx;
    this->HeavyData.Extent.Min[1] = miny;
    this->HeavyData.Extent.Min[2] = minz;
    this->HeavyData.Extent.Max[0] = maxx;
    this->HeavyData.Extent.Max[1] = maxy;
    this->HeavyData.Extent.Max[2] = maxz;
    }

  const dax::internal::StructureUniformGrid& GetHeavyData() const
    { return this->HeavyData; }

protected:
  dax::internal::StructureUniformGrid HeavyData;

private:
  daxDisableCopyMacro(DataArrayStructuredPoints)
};

}}
#endif
