/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_ArrayHandle_h
#define __dax_cont_ArrayHandle_h

#include <dax/Types.h>

#include <dax/internal/DataArray.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/IteratorContainer.h>
#include <dax/cont/internal/IteratorPolymorphic.h>

#include <boost/smart_ptr/shared_ptr.hpp>

#include <assert.h>

namespace dax {
namespace cont {
namespace internal {
class ArrayHandleHelper;
} //internal

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
template<typename T, class DeviceAdapter = DAX_DEFAULT_DEVICE_ADAPTER>
class ArrayHandle
{
public:
  typedef T ValueType;

  /// The default constructor creates a worthless handle to an invalid,
  /// zero-length array. It's mostly here to prevent compile errors when
  /// declaring variables.
  ///
  ArrayHandle() : Internals(new InternalStruct) {
    this->Internals->Synchronized = false;
    this->Internals->NumberOfEntries = 0;
  }

  /// Creates an ArrayHandle that manages data only in the execution
  /// environment. This type of ArrayHandle is good for intermediate arrays
  /// that never have to be accessed or stored outside of the execution
  /// environment.
  ///
  ArrayHandle(dax::Id numEntries) : Internals(new InternalStruct) {
    this->Internals->Synchronized = false;
    this->Internals->NumberOfEntries = numEntries;
  }

  /// Creates an ArrayHandle that manages the data pointed to by the iterators
  /// and any supplemental execution environment memory.
  ///
  template<class IteratorType>
  ArrayHandle(IteratorType begin, IteratorType end)
    : Internals(new InternalStruct(begin, end)) {
    this->Internals->Synchronized = false;
    this->Internals->NumberOfEntries
        = this->Internals->ControlArray.GetNumberOfEntries();
  }

  /// Return the number of entries managed by this handle.
  ///
  dax::Id GetNumberOfEntries() const {return this->Internals->NumberOfEntries;}

  /// True if the control array is still considered valid. When valid,
  /// operations on this array handle should move data to or from that memory
  /// depending on the nature (input or output) of the operation.
  ///
  bool IsControlArrayValid() {return this->Internals->ControlArray.IsValid();}

  /// Marks the iterators passed into the constructor (if there were any) as
  /// invalid. This invalidate is propagated to any copies of the ArrayHandle.
  ///
  void InvalidateControlArray() { this->Internals->ControlArray.Invalidate(); }

  /// Releases any resources used for accessing this data in the execution
  /// environment (such as an array allocation on a computation device). This
  /// release is propagaged to all copies of the ArrayHandle (which all manage
  /// the same data).
  ///
  void ReleaseExecutionResources() {
    this->Internals->ExecutionArray.ReleaseResources();
  }

  /// Returns true if the managed control and execution arrays have the same
  /// data. If there is no control array, then returns true if an execution
  /// result was placed in the execution array. This is not actually checked
  /// (there is no way to do so). Rather, the ReadyAsInput, ReadyAsOutput, and
  /// CompleteAsOutput methods manage synchronization and the synchronization
  /// is assumed to be maintained unless MarkAsUnsynchronized is called, which
  /// should be called if, for example, the control array is modified.
  ///
  bool IsSynchronized() { return this->Internals->Synchronized; }

  /// Call this method if the control and execution data becomes out of sync
  /// (such as if the control data is modified).
  void MarkAsUnsynchronized() { this->Internals->Synchronized = false; }

  /// Allocates the execution array and copies the data in the control array as
  /// necessary.  Returns the array to use in the execution environment.
  ///
  dax::internal::DataArray<ValueType> ReadyAsInput() {
    if (!this->IsSynchronized())
      {
      assert(this->IsControlArrayValid());
      this->Internals->ExecutionArray.Allocate(this->GetNumberOfEntries());
      this->Internals->ExecutionArray.CopyFromControlToExecution(
            this->Internals->ControlArray);
      this->Internals->Synchronized = true;
      }
    return this->Internals->ExecutionArray.GetExecutionArray();
  }

  /// Alloates the execution array as necessary and marks as unsynchronized.
  /// Returns the array to use in the execution environment.
  ///
  dax::internal::DataArray<ValueType> ReadyAsOutput() {
    this->Internals->ExecutionArray.Allocate(this->GetNumberOfEntries());
    this->Internals->Synchronized = false;
    return this->Internals->ExecutionArray.GetExecutionArray();
  }

  /// Completes recording the results of an operation by copying from the
  /// execution environment to the control environment and marking as
  /// synchronized.
  ///
  void CompleteAsOutput() {
    if (this->IsControlArrayValid())
      {
      this->Internals->ExecutionArray.CopyFromExecutionToControl(
            this->Internals->ControlArray);
      }
    else
      {
      // Do nothing.  There is no array because we don't need data.
      }
    this->Internals->Synchronized = true;
  }

  /// Returns the raw execution array. This doesn't verify
  /// any level of synchronization, so the caller must first
  /// make sure the array is of the correct size and is allocated
  const typename DeviceAdapter::template ArrayContainerExecution<ValueType>&
    GetExecutionArray() const
    {
    return this->Internals->ExecutionArray;
    }

private:
  struct InternalStruct {
    dax::cont::internal::IteratorContainer<
      dax::cont::internal::IteratorPolymorphic<ValueType> > ControlArray;

    typename DeviceAdapter::template ArrayContainerExecution<ValueType>
        ExecutionArray;

    bool Synchronized;

    dax::Id NumberOfEntries;

    InternalStruct() { }
    InternalStruct(
        dax::cont::internal::IteratorPolymorphic<ValueType> beginControl,
        dax::cont::internal::IteratorPolymorphic<ValueType> endControl)
      : ControlArray(beginControl, endControl) { }
  };

  boost::shared_ptr<InternalStruct> Internals;

  //make the converter class a friend class
  friend class ArrayHandleHelper;
};

}
}

#endif //__dax_cont_ArrayHandle_h
