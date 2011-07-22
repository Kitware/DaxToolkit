/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxDataArrayIrregular_h
#define __daxDataArrayIrregular_h

#include "daxDataArray.h"
#include "daxTypes.h"
#include <thrust/host_vector.h>

/// daxDataArrayIrregular is the abstract superclass for data array object containing
/// numeric data.
template <class T>
class daxDataArrayIrregular : public daxDataArray
{
public:
  daxDataArrayIrregular();
  virtual ~daxDataArrayIrregular();
  daxTypeMacro(daxDataArrayIrregular, daxDataArray);

  /// Get/Set the number of tuples. Not that the internal data-space will be
  /// resized when the number of tuples is changed.
  void SetNumberOfTuples(size_t num_tuples)
    { this->HeavyData.resize(num_tuples); }
  size_t GetNumberOfTuples() const
    { return this->HeavyData.size(); }

  /// Get/Set value.
  void Set(DaxId index, const T& value)
    { this->HeavyData[index] = value; }
  const T& Get(DaxId index) const
    { return this->HeavyData[index]; }

  /// Called to convert the array to a DaxDataArray which can be passed the
  /// Execution environment.
  virtual bool Convert(DaxDataArray* array);

protected:
  thrust::host_vector<T> HeavyData;
  
private:
  daxDisableCopyMacro(daxDataArrayIrregular)
};

///daxDefinePtrMacro(daxDataArrayIrregular)

#include "daxDataArrayIrregular.txx"

typedef daxDataArrayIrregular<DaxScalar> daxDataArrayScalar;
typedef daxDataArrayIrregular<DaxVector3> daxDataArrayVector3;
typedef daxDataArrayIrregular<DaxVector4> daxDataArrayVector4;

/// declares daxDataArrayPtr
daxDefinePtrMacro(daxDataArrayScalar);
daxDefinePtrMacro(daxDataArrayVector3);
daxDefinePtrMacro(daxDataArrayVector4);

#endif
