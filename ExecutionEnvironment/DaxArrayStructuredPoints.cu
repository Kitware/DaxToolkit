/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __DaxArrayStructuredPoints_h
#define __DaxArrayStructuredPoints_h

#include "DaxArray.cu"

/// DaxArrayStructuredPoints is used for point-coordinates for an uniform grid
/// (vtkImageData).
class DaxArrayStructuredPoints : public DaxArray
{
  SUPERCLASS(DaxArray);
protected:
  struct MetadataType {
    float3 Origin;
    float3 Spacing;
    int3 ExtentMin;
    int3 ExtentMax;
  } Metadata;

public:
  __host__ DaxArrayStructuredPoints() 
    {
    this->Type = STRUCTURED_POINTS;
    this->Metadata.Origin = make_float3(0.0f, 0.0f, 0.0f);
    this->Metadata.Spacing = make_float3(1.0f, 1.0f, 1.0f);
    this->Metadata.ExtentMin = make_int3(0, 0, 0);
    this->Metadata.ExtentMax = make_int3(0, 0, 0);
    }

  __host__ void SetExtent(int xmin, int xmax,
    int ymin, int ymax, int zmin, int zmax)
    {
    this->Metadata.ExtentMin.x = xmin;
    this->Metadata.ExtentMin.y = ymin;
    this->Metadata.ExtentMin.z = zmin;

    this->Metadata.ExtentMax.x = xmax;
    this->Metadata.ExtentMax.y = ymax;
    this->Metadata.ExtentMax.z = zmax;
    }

  __host__ void SetSpacing(float x, float y, float z)
    {
    this->Metadata.Spacing.x = x;
    this->Metadata.Spacing.y = y;
    this->Metadata.Spacing.z = z;
    }

  __host__ void SetOrigin(float x, float y, float z)
    {
    this->Metadata.Origin.x = x;
    this->Metadata.Origin.y = y;
    this->Metadata.Origin.z = z;
    }

  __host__ void Allocate()
    {
    assert(this->OnDevice == false);
    this->Superclass::Allocate(sizeof(Metadata));
    memcpy(this->RawData, &this->Metadata, sizeof(Metadata)); 
    }

protected:
  friend class DaxArrayGetterTraits;

  __device__ static DaxVector3 GetVector3(const DaxWork& work, const DaxArray& array)
    {
    MetadataType* metadata = reinterpret_cast<MetadataType*>(array.RawData);

    DaxId flat_id = work.GetItem();

    // given the flat_id, what is the ijk value?
    int3 dims;
    dims.x = metadata->ExtentMax.x - metadata->ExtentMin.x + 1;
    dims.y = metadata->ExtentMax.y - metadata->ExtentMin.y + 1;
    dims.z = metadata->ExtentMax.z - metadata->ExtentMin.z + 1;

    int3 point_ijk;
    point_ijk.x = flat_id % dims.x;
    point_ijk.y = (flat_id / dims.x)  % dims.y;
    point_ijk.z = flat_id / (dims.x * dims.y);

    DaxVector3 point;
    point.x = metadata->Origin.x + (point_ijk.x + metadata->ExtentMin.x) * metadata->Spacing.x;
    point.y = metadata->Origin.y + (point_ijk.y + metadata->ExtentMin.y) * metadata->Spacing.y;
    point.z = metadata->Origin.z + (point_ijk.z + metadata->ExtentMin.z) * metadata->Spacing.z;
    return point;
    }
};

#endif
