/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxImageData_h
#define __daxImageData_h

#include "daxDataObject.h"

/// daxImageData represents a uniform-rectilinear grid with data-values at each
/// grid point.
class daxImageData : public daxDataObject
{
public:
  daxImageData();
  virtual ~daxImageData();
  daxTypeMacro(daxImageData, daxDataObject);

  /// -------------------------------------------------------------------------
  /// Superclass API
  /// -------------------------------------------------------------------------

  /// API to get the raw data pointer and its size for reading or writing.
  virtual const void* GetDataPointer(const char* data_array_name) const;

  /// API to get the raw data pointer and its size for reading or writing.
  virtual size_t GetDataSize(const char* data_array_name) const;

  /// API to get the raw data pointer and its size for reading or writing.
  virtual void* GetWriteDataPointer(const char* data_array_name);

  /// OpenCL specific API to get the data-implementation code for the device.
  virtual const char* GetCode() const;

  /// These are used to obtain the host data-structure for \c opaque_data_type
  /// instance that's passed to the OpenCL kernel.
  virtual const void* GetOpaqueDataPointer() const;

  /// These are used to obtain the host data-structure for \c opaque_data_type
  /// instance that's passed to the OpenCL kernel.
  virtual size_t GetOpaqueDataSize() const;

  /// -------------------------------------------------------------------------
  /// daxImageData API
  /// -------------------------------------------------------------------------

  /// Get the data-values at each grid point. For now we do not support
  /// named arrays.
  float* GetData()
    { return this->Data; }
  const float* GetData() const
    { return this->Data; }

  float* GetDataPointer(int x, int y, int z);

  /// Get/Set the dimensions of the grid. Changing the dimensions will result in
  /// reallocation of the internal buffer.
  void SetDimensions(int x, int y, int z);
  void SetDimensions(int xyz[3])
    { this->SetDimensions(xyz[0], xyz[1], xyz[2]); }
  const int* GetDimensions() const;

  /// Get/Set the number of components. Changing the number of components will
  //result in reallocation of the internal buffer.
  void SetNumberOfComponents(int numcomps);
  int GetNumberOfComponents() const
    {return this->NumberOfComponents; }

protected:
  int NumberOfComponents;
  int Dimensions[3];
  float* Data;

private:
  daxDisableCopyMacro(daxImageData);
  struct OpaqueDataType;
  OpaqueDataType* OpaqueDataPointer;
};

/// declares daxImageDataPtr.
daxDefinePtrMacro(daxImageData);
#endif
