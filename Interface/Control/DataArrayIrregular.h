/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_DataArrayIrregular_h
#define __dax_cont_DataArrayIrregular_h

#include "Interface/Control/DataArray.h"
#include "BasicTypes/Common/Types.h"

#include <vector>

namespace dax { namespace cont {

/// dax::cont::DataArrayIrregular is the abstract superclass for data array object containing
/// numeric data.
template <class T>
class DataArrayIrregular : public dax::cont::DataArray
{
public:
  DataArrayIrregular();
  virtual ~DataArrayIrregular();
  daxTypeMacro(DataArrayIrregular, dax::cont::DataArray);

  /// Get/Set the number of tuples. Not that the internal data-space will be
  /// resized when the number of tuples is changed.
  void SetNumberOfTuples(size_t num_tuples)
    { this->HeavyData.resize(num_tuples); }
  size_t GetNumberOfTuples() const
    { return this->HeavyData.size(); }

  /// Get/Set value.
  void Set(dax::Id index, const T& value)
    { this->HeavyData[index] = value; }
  const T& Get(dax::Id index) const
    { return this->HeavyData[index]; }

  const std::vector<T>& GetHeavyData() const
    { return this->HeavyData; }
  std::vector<T>& GetHeavyData()
    { return this->HeavyData; }

protected:
  std::vector<T> HeavyData;
  
private:
  daxDisableCopyMacro(DataArrayIrregular)
};

///daxDefinePtrMacro(daxDataArrayIrregular)

#include "Interface/Control/DataArrayIrregular.txx"

typedef dax::cont::DataArrayIrregular<dax::Scalar> DataArrayScalar;
typedef dax::cont::DataArrayIrregular<dax::Vector3> DataArrayVector3;
typedef dax::cont::DataArrayIrregular<dax::Vector4> DataArrayVector4;

/// declares daxDataArrayPtr
daxDefinePtrMacro(DataArrayScalar);
daxDefinePtrMacro(DataArrayVector3);
daxDefinePtrMacro(DataArrayVector4);

}}

#endif
