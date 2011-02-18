/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxImageData.h"

#include "CoreImageData.cl.h"

//-----------------------------------------------------------------------------
struct daxImageData::OpaqueDataType
{
  int Extents[6];
  float Spacing[3];
} __attribute__( (__packed__) );

//-----------------------------------------------------------------------------
daxImageData::daxImageData()
{
  this->NumberOfComponents = 1;
  this->Dimensions[0] = this->Dimensions[1] = this->Dimensions[2] = 0;
  this->Data = NULL;
  this->OpaqueDataPointer = new OpaqueDataType();
}

//-----------------------------------------------------------------------------
daxImageData::~daxImageData()
{
  delete [] this->Data;
  this->Data = NULL;
  delete this->OpaqueDataPointer;
  this->OpaqueDataPointer = NULL;
}

//-----------------------------------------------------------------------------
const void* daxImageData::GetOpaqueDataPointer() const
{
  this->OpaqueDataPointer->Extents[0] = 0;
  this->OpaqueDataPointer->Extents[1] = this->Dimensions[0] - 1;
  this->OpaqueDataPointer->Extents[2] = 0;
  this->OpaqueDataPointer->Extents[3] = this->Dimensions[1] - 1;
  this->OpaqueDataPointer->Extents[4] = 0;
  this->OpaqueDataPointer->Extents[5] = this->Dimensions[2] - 1;
  this->OpaqueDataPointer->Spacing[0] = this->OpaqueDataPointer->Spacing[1] =
    this->OpaqueDataPointer->Spacing[2] = 1.0;
  return this->OpaqueDataPointer;
}

//-----------------------------------------------------------------------------
size_t daxImageData::GetOpaqueDataSize() const
{
  return sizeof(daxImageData::OpaqueDataType);
}

//-----------------------------------------------------------------------------
void daxImageData::SetDimensions(int x, int y, int z)
{
  if (this->Dimensions[0] != x || this->Dimensions[1] != y ||
    this->Dimensions[2] != z)
    {
    delete []this->Data;
    this->Data = new float[x*y*z * this->NumberOfComponents];
    this->Dimensions[0] = x;
    this->Dimensions[1] = y;
    this->Dimensions[2] = z;
    }
}

//-----------------------------------------------------------------------------
void daxImageData::SetNumberOfComponents(int num)
{
  if (this->NumberOfComponents != num)
    {
    delete []this->Data;
    this->Data = new
      float[this->Dimensions[0]*this->Dimensions[1]*this->Dimensions[2]*num];
    this->NumberOfComponents = num;
    }
}

//-----------------------------------------------------------------------------
float* daxImageData::GetDataPointer(int x, int y, int z)
{
  int ijk[3] = { x, y, z};
  int dim[3] = { this->Dimensions[0], this->Dimensions[1], this->Dimensions[2]};
  return this->Data + ((ijk[2]*dim[1] + ijk[1])*dim[0] + ijk[0]);
}

//-----------------------------------------------------------------------------
const int* daxImageData::GetDimensions() const
{
  return this->Dimensions;
}

//-----------------------------------------------------------------------------
const void* daxImageData::GetDataPointer(const char* data_array_name) const
{
  return this->GetData();
}

//-----------------------------------------------------------------------------
size_t daxImageData::GetDataSize(const char* data_array_name) const
{
  return this->GetDimensions()[0] * this->GetDimensions()[1] *
    this->GetDimensions()[2] * this->GetNumberOfComponents() * sizeof(float);
}

//-----------------------------------------------------------------------------
void* daxImageData::GetWriteDataPointer(const char* data_array_name)
{
  return this->GetData();
}

//-----------------------------------------------------------------------------
const char* daxImageData::GetCode() const
{
  return daxHeaderString_CoreImageData;
}
