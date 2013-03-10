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

#include <dax/Types.h>

#include <dax/cont/ArrayContainerControl.h>
#include <dax/cont/Assert.h>
#include <dax/cont/ErrorControlBadValue.h>
#include <dax/cont/internal/ArrayTransfer.h>
#include <dax/cont/internal/DeviceAdapterTag.h>

#include <boost/concept_check.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>

#include <vector>

namespace dax {
namespace cont {

// Forward declaration
namespace internal { class ArrayHandleAccess; }

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
template<
    typename T,
    class ArrayContainerControlTag = DAX_DEFAULT_ARRAY_CONTAINER_CONTROL_TAG,
    class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class ArrayHandle
{
private:
  typedef dax::cont::internal::ArrayContainerControl<T,ArrayContainerControlTag>
      ArrayContainerControlType;
  typedef dax::cont::internal
      ::ArrayTransfer<T,ArrayContainerControlTag,DeviceAdapterTag>
      ArrayTransferType;
public:
  typedef T ValueType;
  typedef typename ArrayContainerControlType::PortalType PortalControl;
  typedef typename ArrayContainerControlType::PortalConstType
      PortalConstControl;
  typedef typename ArrayTransferType::PortalExecution PortalExecution;
  typedef typename ArrayTransferType::PortalConstExecution PortalConstExecution;

  /// Constructs an empty ArrayHandle. Typically used for output or
  /// intermediate arrays that will be filled by a Dax algorithm.
  ///
  DAX_CONT_EXPORT ArrayHandle() : Internals(new InternalStruct)
  {
    this->Internals->UserPortalValid = false;
    this->Internals->ControlArrayValid = false;
    this->Internals->ExecutionArrayValid = false;
  }

  /// Constructs an ArrayHandle pointing to the data in the given array portal.
  ///
  DAX_CONT_EXPORT ArrayHandle(PortalConstControl userData)
    : Internals(new InternalStruct)
  {
    this->Internals->UserPortal = userData;
    this->Internals->UserPortalValid = true;

    this->Internals->ControlArrayValid = false;
    this->Internals->ExecutionArrayValid = false;
  }

  /// Get the array portal of the control array.
  ///
  DAX_CONT_EXPORT PortalControl GetPortalControl()
  {
    this->SyncControlArray();
    if (this->Internals->UserPortalValid)
      {
      throw dax::cont::ErrorControlBadValue(
            "ArrayHandle has a read-only control portal.");
      }
    else if (this->Internals->ControlArrayValid)
      {
      // If the user writes into the iterator we return, then the execution
      // array will become invalid. Play it safe and release the execution
      // resources. (Use the const version to preserve the execution array.)
      this->ReleaseResourcesExecution();
      return this->Internals->ControlArray.GetPortal();
      }
    else
      {
      throw dax::cont::ErrorControlBadValue("ArrayHandle contains no data.");
      }
  }

  /// Get the array portal of the control array.
  ///
  DAX_CONT_EXPORT PortalConstControl GetPortalConstControl() const
  {
    this->SyncControlArray();
    if (this->Internals->UserPortalValid)
      {
      return this->Internals->UserPortal;
      }
    else if (this->Internals->ControlArrayValid)
      {
      return this->Internals->ControlArray.GetPortalConst();
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
    if (this->Internals->UserPortalValid)
      {
      return this->Internals->UserPortal.GetNumberOfValues();
      }
    else if (this->Internals->ControlArrayValid)
      {
      return this->Internals->ControlArray.GetNumberOfValues();
      }
    else if (this->Internals->ExecutionArrayValid)
      {
      return
          this->Internals->ExecutionArray.GetNumberOfValues();
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
    BOOST_CONCEPT_ASSERT((boost::OutputIterator<IteratorType, ValueType>));
    BOOST_CONCEPT_ASSERT((boost::ForwardIterator<IteratorType>));
    if (this->Internals->ExecutionArrayValid)
      {
      this->Internals->ExecutionArray.CopyInto(dest);
      }
    else
      {
      PortalConstControl portal = this->GetPortalConstControl();
      std::copy(portal.GetIteratorBegin(), portal.GetIteratorEnd(), dest);
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
      if (this->Internals->UserPortalValid)
        {
        throw dax::cont::ErrorControlBadValue(
              "ArrayHandle has a read-only control portal.");
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
    this->Internals->UserPortalValid = false;

    if (this->Internals->ControlArrayValid)
      {
      this->Internals->ControlArray.ReleaseResources();
      this->Internals->ControlArrayValid = false;
      }
  }

  /// Prepares this array to be used as an input to an operation in the
  /// execution environment. If necessary, copies data to the execution
  /// environment. Can throw an exception if this array does not yet contain
  /// any data. Returns a portal that can be used in code running in the
  /// execution environment.
  ///
  DAX_CONT_EXPORT
  PortalConstExecution PrepareForInput() const
  {
    if (this->Internals->ExecutionArrayValid)
      {
      // Nothing to do, data already loaded.
      }
    else if (this->Internals->UserPortalValid)
      {
      DAX_ASSERT_CONT(!this->Internals->ControlArrayValid);
      this->Internals->ExecutionArray.LoadDataForInput(
            this->Internals->UserPortal);
      this->Internals->ExecutionArrayValid = true;
      }
    else if (this->Internals->ControlArrayValid)
      {
      this->Internals->ExecutionArray.LoadDataForInput(
            this->Internals->ControlArray.GetPortalConst());
      this->Internals->ExecutionArrayValid = true;
      }
    else
      {
      throw dax::cont::ErrorControlBadValue(
            "ArrayHandle has no data when PrepareForInput called.");
      }
    return this->Internals->ExecutionArray.GetPortalConstExecution();
  }

  /// Prepares (allocates) this array to be used as an output from an operation
  /// in the execution environment. The internal state of this class is set to
  /// have valid data in the execution array with the assumption that the array
  /// will be filled soon (i.e. before any other methods of this object are
  /// called). Returns a portal that can be used in code running in the
  /// execution environment.
  ///
  DAX_CONT_EXPORT
  PortalExecution PrepareForOutput(dax::Id numberOfValues)
  {
    // Invalidate any control arrays.
    // Should the control array resource be released? Probably not a good
    // idea when shared with execution.
    this->Internals->UserPortalValid = false;
    this->Internals->ControlArrayValid = false;

    this->Internals->ExecutionArray.AllocateArrayForOutput(
          this->Internals->ControlArray, numberOfValues);

    // We are assuming that the calling code will fill the array using the
    // iterators we are returning, so go ahead and mark the execution array as
    // having valid data. (A previous version of this class had a separate call
    // to mark the array as filled, but that was onerous to call at the the
    // right time and rather pointless since it is basically always the case
    // that the array is going to be filled before anything else. In this
    // implementation the only access to the array is through the iterators
    // returned from this method, so you would have to work to invalidate this
    // assumption anyway.)
    this->Internals->ExecutionArrayValid = true;

    return this->Internals->ExecutionArray.GetPortalExecution();
  }

  /// Prepares this array to be used in an in-place operation (both as input
  /// and output) in the execution environment. If necessary, copies data to
  /// the execution environment. Can throw an exception if this array does not
  /// yet contain any data. Returns a portal that can be used in code running
  /// in the execution environment.
  ///
  DAX_CONT_EXPORT
  PortalExecution PrepareForInPlace()
  {
    if (this->Internals->UserPortalValid)
      {
      throw dax::cont::ErrorControlBadValue(
            "In place execution cannot be used with an ArrayHandle that has "
            "user arrays because this might write data back into user space "
            "unexpectedly.  Copy the data to a new array first.");
      }

    // This code is similar to PrepareForInput except that we have to give a
    // writable portal instead of the const portal to the execution array
    // manager so that the data can (potentially) be written to.
    if (this->Internals->ExecutionArrayValid)
      {
      // Nothing to do, data already loaded.
      }
    else if (this->Internals->ControlArrayValid)
      {
      this->Internals->ExecutionArray.LoadDataForInPlace(
            this->Internals->ControlArray);
      this->Internals->ExecutionArrayValid = true;
      }
    else
      {
      throw dax::cont::ErrorControlBadValue(
            "ArrayHandle has no data when PrepareForInput called.");
      }

    // Invalidate any control arrays since their data will become invalid when
    // the execution data is overwritten. Don't actually release the control
    // array. It may be shared as the execution array.
    this->Internals->ControlArrayValid = false;

    return this->Internals->ExecutionArray.GetPortalExecution();
  }

private:
  struct InternalStruct {
    PortalConstControl UserPortal;
    bool UserPortalValid;

    ArrayContainerControlType ControlArray;
    bool ControlArrayValid;

    ArrayTransferType ExecutionArray;
    bool ExecutionArrayValid;
  };

  /// Synchronizes the control array with the execution array. If either the
  /// user array or control array is already valid, this method does nothing
  /// (because the data is already available in the control environment).
  /// Although the internal state of this class can change, the method is
  /// declared const because logically the data does not.
  ///
  DAX_CONT_EXPORT void SyncControlArray() const
  {
    if (   !this->Internals->UserPortalValid
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
      DAX_ASSERT_CONT(!this->Internals->UserPortalValid
                      || !this->Internals->ControlArrayValid);
      // Nothing to do.
      }
  }

  boost::shared_ptr<InternalStruct> Internals;
};

/// A convenience function for creating an ArrayHandle from a standard C
/// array.  Unless properly specialized, this only works with container types
/// that use an array portal that accepts a pair of pointers to signify the
/// beginning and end of the array.
///
template<typename T, class ArrayContainerControlTag, class DeviceAdapterTag>
DAX_CONT_EXPORT
dax::cont::ArrayHandle<T, ArrayContainerControlTag, DeviceAdapterTag>
make_ArrayHandle(const T *array,
                 dax::Id length,
                 ArrayContainerControlTag,
                 DeviceAdapterTag)
{
  typedef dax::cont::ArrayHandle<T, ArrayContainerControlTag, DeviceAdapterTag>
      ArrayHandleType;
  typedef typename ArrayHandleType::PortalConstControl PortalType;
  return ArrayHandleType(PortalType(array, array+length));
}
template<typename T, class ArrayContainerControlTag>
DAX_CONT_EXPORT
dax::cont::ArrayHandle<T, ArrayContainerControlTag, DAX_DEFAULT_DEVICE_ADAPTER_TAG>
make_ArrayHandle(const T *array, dax::Id length, ArrayContainerControlTag)
{
  return make_ArrayHandle(array,
                          length,
                          ArrayContainerControlTag(),
                          DAX_DEFAULT_DEVICE_ADAPTER_TAG());
}
template<typename T>
DAX_CONT_EXPORT
dax::cont::ArrayHandle<
     T, DAX_DEFAULT_ARRAY_CONTAINER_CONTROL_TAG, DAX_DEFAULT_DEVICE_ADAPTER_TAG>
make_ArrayHandle(const T *array, dax::Id length)
{
  return make_ArrayHandle(array,
                          length,
                          DAX_DEFAULT_ARRAY_CONTAINER_CONTROL_TAG(),
                          DAX_DEFAULT_DEVICE_ADAPTER_TAG());
}

/// A convenience function for creating an ArrayHandle from an std::vector.
/// Unless properly specialized, this only works with container types that use
/// an array portal that accepts a pair of pointers to signify the beginning
/// and end of the array.
///
template<typename T,
         typename Allocator,
         class ArrayContainerControlTag,
         class DeviceAdapterTag>
DAX_CONT_EXPORT
dax::cont::ArrayHandle<T, ArrayContainerControlTag, DeviceAdapterTag>
make_ArrayHandle(const std::vector<T,Allocator> &array,
                 ArrayContainerControlTag,
                 DeviceAdapterTag)
{
  typedef dax::cont::ArrayHandle<T, ArrayContainerControlTag, DeviceAdapterTag>
      ArrayHandleType;
  typedef typename ArrayHandleType::PortalConstControl PortalType;
  return ArrayHandleType(PortalType(&array.front(), &array.back() + 1));
}
template<typename T,
         typename Allocator,
         class ArrayContainerControlTag>
DAX_CONT_EXPORT
dax::cont::ArrayHandle<T, ArrayContainerControlTag, DAX_DEFAULT_DEVICE_ADAPTER_TAG>
make_ArrayHandle(const std::vector<T,Allocator> &array, ArrayContainerControlTag)
{
  return make_ArrayHandle(array,
                          ArrayContainerControlTag(),
                          DAX_DEFAULT_DEVICE_ADAPTER_TAG());
}
template<typename T, typename Allocator>
DAX_CONT_EXPORT
dax::cont::ArrayHandle<
    T, DAX_DEFAULT_ARRAY_CONTAINER_CONTROL_TAG, DAX_DEFAULT_DEVICE_ADAPTER_TAG>
make_ArrayHandle(const std::vector<T,Allocator> &array)
{
  return make_ArrayHandle(array,
                          DAX_DEFAULT_ARRAY_CONTAINER_CONTROL_TAG(),
                          DAX_DEFAULT_DEVICE_ADAPTER_TAG());
}

}
}

//to simplify the user experience we bring in all different types of array
//handles when you include array handle
#include <dax/cont/ArrayHandleCounting.h>
#include <dax/cont/ArrayHandleConstantValue.h>
#endif //__dax_cont_ArrayHandle_h
