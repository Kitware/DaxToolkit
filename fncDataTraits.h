/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// This file defines different traits that should be implemented by concrete
// data types.

template <typename T>
struct fncReadableDataTraits
{
  /// Returns the raw data-pointer.
  static const void* GetDataPointer(const char* data_array_name, const T* data)
    {
    (void)data_array_name;
    (void)data;
    return NULL;
    }

  /// Returns the buffer size in bytes.
  static size_t GetDataSize(const char* data_array_name, const T* data)
    {
    (void)data_array_name;
    (void)data;
    return 0;
    }
};


template <typename T>
struct fncWriteableDataTraits
{
};


template <typename T>
struct fncOpenCLTraits
{
  /// Returns the OpenCL code defining different datatypes and iterator
  /// functions.
  static std::string GetCode()
    {
    return std::string();
    }
};
