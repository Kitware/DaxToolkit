/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_ArrayHandle_h
#define __dax_cont_ArrayHandle_h

#include <dax/Types.h>

#include <dax/cont/internal/IteratorContainer.h>
#include <dax/cont/internal/IteratorPolymorphic.h>

#include <boost/smart_ptr/shared_ptr.hpp>

namespace dax {
namespace cont {

/// Manages an array-worth of data. Typically this data holds field data. The
/// array handle optionally contains a reference to user managed data, with
/// which it can read input data or write results data. The array handle also
/// manages any memory needed to access the data from the execution
/// environment.
///
/// An ArrayHandle can be created without pointing to any actual user data. In
/// this case, ArrayHandle will only maintain an array accessed from the
/// execution environment, and this data will not be accessible from the
/// control environment. This is helpful for intermediate results that are not
/// themselves important and saves any cost with transferring the data to the
/// control environment.
///
/// If the array previously pointed to by an ArrayHandle becomes invalid (for
/// example, the data becomes invalid or the memory is freed), it can be marked
/// as invalid in the ArrayHandle, which will serve as a flag to no longer use
/// that memory. This invalid mark will propagate to all copies of the
/// ArrayHandle instance.
///
/// Any memory created for the execution environment will remain around in case
/// it is needed again.
///
template<typename T>
class ArrayHandle
{
public:
  typedef T ValueType;

  /// The default constructor creates a worthless handle to an invalid,
  /// zero-length array. It's mostly here to prevent compile errors when
  /// declaring variables.
  ///
  ArrayHandle() : ControlArray(new ControlArrayType), NumberOfEntries(0) { }

  /// Creates an ArrayHandle that manages data only in the execution
  /// environment. This type of ArrayHandle is good for intermediate arrays
  /// that never have to be accessed or stored outside of the execution
  /// environment.
  ///
  ArrayHandle(dax::Id numEntries)
    : ControlArray(new ControlArrayType), NumberOfEntries(numEntries) { }

  /// Creates an ArrayHandle that manages the data pointed to by the iterators
  /// and any supplemental execution environment memory.
  ///
  template<class IteratorType>
  ArrayHandle(IteratorType begin, IteratorType end)
    : ControlArray(new ControlArrayType(begin, end)) {
    this->NumberOfEntries = this->ControlArray->GetNumberOfEntries();
  }

  /// Return the number of entries managed by this handle.
  ///
  dax::Id GetNumberOfEntries() const { return this->NumberOfEntries; }

  /// True if the control array is still considered valid. When valid,
  /// operations on this array handle should move data to or from that memory
  /// depending on the nature (input or output) of the operation.
  ///
  bool IsControlArrayValid() { return this->ControlArray->IsValid(); }

  /// Marks the iterators passed into the constructor (if there were any) as
  /// invalid. This invalidate is propagated to any copies of the ArrayHandle.
  ///
  void InvalidateControlArray() { this->ControlArray->Invalidate(); }

  /// Releases any resources used for accessing this data in the execution
  /// environment (such as an array allocation on a computation device). This
  /// release is propagaged to all copies of the ArrayHandle (which all manage
  /// the same data).
  ///
  void ReleaseExecutionResources() {
    // Implement this.
  }

private:
  typedef dax::cont::internal::IteratorContainer<
    dax::cont::internal::IteratorPolymorphic<ValueType> > ControlArrayType;

  boost::shared_ptr<ControlArrayType> ControlArray;

  dax::Id NumberOfEntries;
};

}
}

#endif //__dax_cont_ArrayHandle_h
