//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_cont_ArrayHandle_h
#define __dax_cont_ArrayHandle_h

// Make a forward declaration of ArrayHandle so that when DeviceAdapter gets
// included its implementations will compile correctly. This chicken-and-egg
// problem could get averted by simply forward declaring
// DAX_DEFAULT_DEVICE_ADAPTER, which is the only reason we need to include
// DeviceAdapter in the first place, but to do that you would need more
// complicated headers so that the macro and prototype was included here and
// then the implementation somewhere later... Well, it's just a lot easier to
// declare a prototype here.
//
namespace dax { namespace cont {
template <typename T,
          template <typename> class ArrayContainerControl,
          class DeviceAdapter>
class ArrayHandle;
} }


#include <dax/Types.h>

#include <dax/cont/ArrayContainerControl.h>
#include <dax/cont/Assert.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/ErrorControlBadValue.h>

#include <boost/smart_ptr/shared_ptr.hpp>

#include <utility>

namespace dax {
namespace cont {

/// \brief Manages an array-worth of data.
///
/// \c ArrayHandle manages as array of data that can be manipulated by Dax
/// algorithms. The \c ArrayHandle may have up to two copies of the array, one
/// for the control environment and one for the execution environment, although
/// depending on the device and how the array is being used, the \c ArrayHandle
/// will only have one copy when possible.
///
/// An ArrayHandle can be constructed one of two ways. Its default construction
/// creates an empty, unallocated array that can later be allocated and filled
/// either by the user or a Dax algorithm. The \c ArrayHandle can also be
/// constructed with iterators to a user's array. In this case the \c
/// ArrayHandle will keep a reference to this array but may drop it if the
/// array is reallocated.
///
/// \c ArrayHandle behaves like a shared smart pointer in that when it is copied
/// each copy holds a reference to the same array.  These copies are reference
/// counted so that when all copies of the \c ArrayHandle are destroyed, any
/// allocated memory is released.
///
template <typename T,
          template <typename> class ArrayContainerControl
              = DAX_DEFAULT_ARRAY_CONTAINER_CONTROL,
          class DeviceAdapter = DAX_DEFAULT_DEVICE_ADAPTER>
class ArrayHandle
{
private:
  typedef ArrayContainerControl<T> ArrayContainerControlType;
  typedef typename DeviceAdapter::template ArrayManagerExecution
      <T, ArrayContainerControl> ArrayManagerExecutionType;
public:
  typedef T ValueType;
  typedef typename ArrayContainerControlType::IteratorType IteratorControl;
  typedef typename ArrayContainerControlType::IteratorConstType
      IteratorConstControl;
  typedef typename ArrayManagerExecutionType::IteratorType IteratorExecution;
  typedef typename ArrayManagerExecutionType::IteratorConstType
      IteratorConstExecution;
  typedef typename DeviceAdapter
      ::template ExecutionAdapter<ArrayContainerControl> ExecutionAdapter;

  /// Constructs an empty ArrayHandle. Typically used for output or
  /// intermediate arrays that will be filled by a Dax algorithm.
  ///
  DAX_CONT_EXPORT ArrayHandle() : Internals(new InternalStruct)
  {
    this->Internals->UserIteratorValid = false;
    this->Internals->ControlArrayValid = false;
    this->Internals->ExecutionArrayValid = false;
  }

  /// Constructs an ArrayHandle pointing to the data in the given iterators.
  ///
  DAX_CONT_EXPORT ArrayHandle(IteratorConstControl userDataBegin,
                              IteratorConstControl userDataEnd)
    : Internals(new InternalStruct)
  {
    this->Internals->UserIteratorBegin = userDataBegin;
    this->Internals->UserIteratorEnd = userDataEnd;
    this->Internals->UserIteratorValid = true;

    this->Internals->ControlArrayValid = false;
    this->Internals->ExecutionArrayValid = false;
  }

  /// The begin iterator of the control array.
  ///
  DAX_CONT_EXPORT IteratorControl GetIteratorControlBegin()
  {
    this->SyncControlArray();
    if (this->Internals->UserIteratorValid)
      {
      throw dax::cont::ErrorControlBadValue(
            "ArrayHandle has a read-only control iterator.");
      }
    else if (this->Internals->ControlArrayValid)
      {
      return this->Internals->ControlArray.GetIteratorBegin();
      }
    else
      {
      throw dax::cont::ErrorControlBadValue("ArrayHandle contains no data.");
      }
  }

  /// The end iterator of the control array.
  ///
  DAX_CONT_EXPORT IteratorControl GetIteratorControlEnd()
  {
    this->SyncControlArray();
    if (this->Internals->UserIteratorValid)
      {
      throw dax::cont::ErrorControlBadValue(
            "ArrayHandle has a read-only control iterator.");
      }
    else if (this->Internals->ControlArrayValid)
      {
      return this->Internals->ControlArray.GetIteratorEnd();
      }
    else
      {
      throw dax::cont::ErrorControlBadValue("ArrayHandle contains no data.");
      }
  }

  /// The begin iterator of the control array.
  ///
  DAX_CONT_EXPORT IteratorConstControl GetIteratorConstControlBegin() const
  {
    this->SyncControlArray();
    if (this->Internals->UserIteratorValid)
      {
      return this->Internals->UserIteratorBegin;
      }
    else if (this->Internals->ControlArrayValid)
      {
      return this->Internals->ControlArray.GetIteratorConstBegin();
      }
    else
      {
      throw dax::cont::ErrorControlBadValue("ArrayHandle contains no data.");
      }
  }

  /// The end iterator of the control array.
  ///
  DAX_CONT_EXPORT IteratorConstControl GetIteratorConstControlEnd() const
  {
    this->SyncControlArray();
    if (this->Internals->UserIteratorValid)
      {
      return this->Internals->UserIteratorEnd;
      }
    else if (this->Internals->ControlArrayValid)
      {
      return this->Internals->ControlArray.GetIteratorConstEnd();
      }
    else
      {
      throw dax::cont::ErrorControlBadValue("ArrayHandle contains no data.");
      }
  }

  /// Returns the number of entries in the array.
  ///
  DAX_CONT_EXPORT dax::Id GetNumberOfValues() const
  {
    if (this->Internals->UserIteratorValid)
      {
      return std::distance(this->Internals->UserIteratorBegin,
                           this->Internals->UserIteratorEnd);
      }
    else if (this->Internals->ControlArrayValid)
      {
      return this->Internals->ControlArray.GetNumberOfValues();
      }
    else if (this->Internals->ExecutionArrayValid)
      {
      return
          std::distance(this->Internals->ExecutionArray.GetIteratorConstBegin(),
                        this->Internals->ExecutionArray.GetIteratorConstEnd());
      }
    else
      {
      return 0;
      }
  }

  /// Copies data into the given iterator for the control environment. This
  /// method can skip copying into an internally managed control array.
  ///
  template <class IteratorType>
  DAX_CONT_EXPORT void CopyInto(IteratorType dest) const
  {
    if (this->Internals->ExecutionArrayValid)
      {
      this->Internals->ExecutionArray.CopyInto(dest);
      }
    else
      {
      std::copy(this->GetIteratorConstControlBegin(),
                this->GetIteratorConstControlEnd(),
                dest);
      }
  }

  /// \brief Reduces the size of the array without changing its values.
  ///
  /// This method allows you to resize the array without reallocating it. The
  /// number of entries in the array is changed to \c numberOfValues. The data
  /// in the array (from indices 0 to \c numberOfValues - 1) are the same, but
  /// \c numberOfValues must be equal or less than the preexisting size
  /// (returned from GetNumberOfValues). That is, this method can only be used
  /// to shorten the array, not lengthen.
  void Shrink(dax::Id numberOfValues)
  {
    dax::Id originalNumberOfValues = this->GetNumberOfValues();

    if (numberOfValues < originalNumberOfValues)
      {
      if (this->Internals->UserIteratorValid)
        {
        this->Internals->UserIteratorEnd = this->Internals->UserIteratorBegin;
        std::advance(this->Internals->UserIteratorEnd, numberOfValues);
        }
      if (this->Internals->ControlArrayValid)
        {
        this->Internals->ControlArray.Shrink(numberOfValues);
        }
      if (this->Internals->ExecutionArrayValid)
        {
        this->Internals->ExecutionArray.Shrink(numberOfValues);
        }
      }
    else if (numberOfValues == originalNumberOfValues)
      {
      // Nothing to do.
      }
    else // numberOfValues > originalNumberOfValues
      {
      throw dax::cont::ErrorControlBadValue(
            "ArrayHandle::Shrink cannot be used to grow array.");
      }

    DAX_ASSERT_CONT(this->GetNumberOfValues() == numberOfValues);
  }

  /// Releases any resources being used in the execution environment (that are
  /// not being shared by the control environment).
  ///
  DAX_CONT_EXPORT void ReleaseResourcesExecution()
  {
    if (this->Internals->ExecutionArrayValid)
      {
      this->Internals->ExecutionArray.ReleaseResources();
      this->Internals->ExecutionArrayValid = false;
      }
  }

  /// Releases all resources in both the control and execution environments.
  ///
  DAX_CONT_EXPORT void ReleaseResources()
  {
    this->ReleaseResourcesExecution();

    // Forget about any user iterators.
    this->Internals->UserIteratorValid = false;

    if (this->Internals->ControlArrayValid)
      {
      this->Internals->ControlArray.ReleaseResources();
      this->Internals->ControlArrayValid = false;
      }
  }

  /// Prepares this array to be used as an input to an operation in the
  /// execution environment. If necessary, copies data to the execution
  /// environment. Can throw an exception if this array does not yet contain
  /// any data. Returns a pair of begin/end iterators that can be used in code
  /// running in the execution environment.
  ///
  DAX_CONT_EXPORT std::pair<IteratorConstExecution, IteratorConstExecution>
  PrepareForInput() const
  {
    if (this->Internals->ExecutionArrayValid)
      {
      // Nothing to do, data already loaded.
      }
    else if (this->Internals->UserIteratorValid)
      {
      DAX_ASSERT_CONT(!this->Internals->ControlArrayValid);
      this->Internals->ExecutionArray.LoadDataForInput(
            this->Internals->UserIteratorBegin,
            this->Internals->UserIteratorEnd);
      this->Internals->ExecutionArrayValid = true;
      }
    else if (this->Internals->ControlArrayValid)
      {
      this->Internals->ExecutionArray.LoadDataForInput(
            this->Internals->ControlArray.GetIteratorConstBegin(),
            this->Internals->ControlArray.GetIteratorConstEnd());
      this->Internals->ExecutionArrayValid = true;
      }
    else
      {
      throw dax::cont::ErrorControlBadValue(
            "ArrayHandle has no data when PrepareForInput called.");
      }
    return std::make_pair(
          this->Internals->ExecutionArray.GetIteratorConstBegin(),
          this->Internals->ExecutionArray.GetIteratorConstEnd());
  }

  /// Prepares (allocates) this array to be used as an output from an operation
  /// in the execution environment. The internal state of this class is set to
  /// have valid data in the execution array with the assumption that the array
  /// will be filled soon (i.e. before any other methods of this object are
  /// called). Returns a pair of begin/end iterators that can be used in code
  /// running in the execution environment.
  ///
  DAX_CONT_EXPORT std::pair<IteratorExecution, IteratorExecution>
  PrepareForOutput(dax::Id numberOfValues)
  {
    // Invalidate any control arrays.
    // Should the control array resource be released? Probably not be a good
    // idea when shared with execution.
    this->Internals->UserIteratorValid = false;
    this->Internals->ControlArrayValid = false;

    this->Internals->ExecutionArray.AllocateArrayForOutput(
          this->Internals->ControlArray, numberOfValues);

    // We are assuming that the calling code will fill the array using the
    // iterators we are returning, so go ahead ad mark the execution array as
    // having valid data. (A previous version of this class had a separate call
    // to mark the array as filled, but that was onerous to call at the the
    // right time and rather pointless since it is basically always the case
    // that the array is going to be filled before anything else. In this
    // implementation the only access to the array is through the iterators
    // returned from this method, so you would have to work to invalidate this
    // assumption anyway.
    this->Internals->ExecutionArrayValid = true;

    return std::make_pair(this->Internals->ExecutionArray.GetIteratorBegin(),
                          this->Internals->ExecutionArray.GetIteratorEnd());
  }

  /// Prepares this array to be used in an in-place operation (both as input
  /// and output) in the execution environment. If necessary, copies data to
  /// the execution environment. Can throw an exception if this array does not
  /// yet contain any data. Returns a a pair of begin/end iterators that can be
  /// used in code running in the execution environment.
  ///
  DAX_CONT_EXPORT std::pair<IteratorExecution, IteratorExecution>
  PrepareForInPlace()
  {
    this->PrepareForInput();

    // Invalidate any control arrays. Don't actually release the control array.
    // It may be shared as the execution array.
    this->Internals->UserIteratorValid = false;
    this->Internals->ControlArrayValid = false;

    return std::make_pair(this->Internals->ExecutionArray.GetIteratorBegin(),
                          this->Internals->ExecutionArray.GetIteratorEnd());
  }

private:
  struct InternalStruct {
    IteratorConstControl UserIteratorBegin;
    IteratorConstControl UserIteratorEnd;
    bool UserIteratorValid;

    ArrayContainerControlType ControlArray;
    bool ControlArrayValid;

    ArrayManagerExecutionType ExecutionArray;
    bool ExecutionArrayValid;
  };

  /// Synchronizes the control array with the execution array. If either the
  /// user array or control array is already valid, this method does nothing
  /// (because the data is already available in the control environment.
  /// Although the internal state of this class can change, the method is
  /// declared const because logically the data does not.
  ///
  DAX_CONT_EXPORT void SyncControlArray() const
  {
    if (   !this->Internals->UserIteratorValid
        && !this->Internals->ControlArrayValid)
      {
      // Need to change some state that does not change the logical state from
      // an external point of view.
      InternalStruct *internals
          = const_cast<InternalStruct*>(this->Internals.get());
      internals->ExecutionArray.RetrieveOutputData(internals->ControlArray);
      internals->ControlArrayValid = true;
      }
    else
      {
      // It should never be the case that both the user and control array are
      // valid.
      DAX_ASSERT_CONT(!this->Internals->UserIteratorValid
                      || !this->Internals->ControlArrayValid);
      // Nothing to do.
      }
  }

  boost::shared_ptr<InternalStruct> Internals;
};

}
}

#endif //__dax_cont_ArrayHandle_h
