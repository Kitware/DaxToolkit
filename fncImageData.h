/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __fncImageData_h
#define __fncImageData_h

#include "fncObject.h"

/// fncImageData represents a uniform-rectilinear grid with data-values at each
/// grid point.
class fncImageData : public fncObject
{
public:
  fncImageData();
  virtual ~fncImageData();
  fncTypeMacro(fncImageData, fncObject);

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

  /// OpaqueDataPointer encapsulates all information about this mesh/grid that
  /// needs to be sent to the device.
  void* GetOpaqueDataPointer() const;
  size_t GetOpaqueDataSize() const;

protected:
  int NumberOfComponents;
  int Dimensions[3];
  float* Data;

private:
  fncDisableCopyMacro(fncImageData);
  struct OpaqueDataType;
  OpaqueDataType* OpaqueDataPointer;
};

/// declares fncImageDataPtr.
fncDefinePtrMacro(fncImageData);

/// Implement data-traits.
#include "fncDataTraits.h"

template <>
struct fncReadableDataTraits<fncImageData>
{
  /// Returns the raw data-pointer.
  static const void* GetDataPointer(const char*, const fncImageData* data)
    { return data->GetData(); }

  /// Returns the buffer size in bytes.
  static size_t GetDataSize(const char*, const fncImageData* data)
    {
    return data->GetDimensions()[0] * data->GetDimensions()[1] *
      data->GetDimensions()[2] * data->GetNumberOfComponents() * sizeof(float);
    }
};

namespace fncImageDataInternals
{
  std::string GetOpenCLCode();
};

template <>
struct fncOpenCLTraits<fncImageData>
{
  /// Returns the OpenCL code defining different datatypes and iterator
  /// functions.
  static std::string GetCode()
    { return fncImageDataInternals::GetOpenCLCode(); }

  /// These are used to obtain the host data-structure for \c opaque_data_type
  /// instance that's passed to the OpenCL kernel.
  static void* GetOpaqueDataPointer(const fncImageData* data)
    { return data->GetOpaqueDataPointer(); }
  static size_t GetOpaqueDataSize(const fncImageData* data)
    { return data->GetOpaqueDataSize(); }
};
#endif
