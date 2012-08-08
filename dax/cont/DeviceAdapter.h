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

#ifndef __dax_cont_DeviceAdapter_h
#define __dax_cont_DeviceAdapter_h

#define DAX_DEVICE_ADAPTER_ERROR     -1
#define DAX_DEVICE_ADAPTER_UNDEFINED  0
#define DAX_DEVICE_ADAPTER_SERIAL     1
#define DAX_DEVICE_ADAPTER_CUDA       2
#define DAX_DEVICE_ADAPTER_OPENMP     3

#ifndef DAX_DEVICE_ADAPTER
#ifdef DAX_CUDA
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_CUDA
#elif defined(DAX_OPENMP) // !DAX_CUDA
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_OPENMP
#else // !DAX_CUDA && !DAX_OPENMP
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_SERIAL
#endif // !DAX_CUDA && !DAX_OPENMP
#endif // DAX_DEVICE_ADAPTER

namespace dax {
namespace cont {

#ifdef DAX_DOXYGEN_ONLY
/// \brief A tag specifying the interface between the control and execution environments.
///
/// A DeviceAdapter tag specifies a set of functions and classes that provide
/// mechanisms to run algorithms on a type of parallel device. The tag
/// DeviceAdapterTag___ does not actually exist. Rather, this documentation is
/// provided to describe the interface for a DeviceAdapter. Loading the
/// dax/cont/DeviceAdapter.h header file will set a default device adapter
/// appropriate for the current compile environment. You can specify the
/// default device adapter by first setting the \c DAX_DEVICE_ADAPTER macro.
/// Valid values for \c DAX_DEVICE_ADAPTER are the following:
///
/// \li \c DAX_DEVICE_ADAPTER_SERIAL Runs all algorithms in serial. Can be
/// helpful for debugging.
/// \li \c DAX_DEVICE_ADAPTER_CUDA Schedules and runs algorithms on a GPU
/// using CUDA.  Must be compiling with a CUDA compiler (nvcc).
/// \li \c DAX_DEVICE_ADAPTER_OPENMP Schedules an algorithm over multiple
/// CPU cores using OpenMP compiler directives.  Must be compiling with an
/// OpenMP-compliant compiler with OpenMP pragmas enabled.
///
/// See the ArrayManagerExecution.h and DeviceAdapterAlgorithm.h files for
/// documentation on all the functions and classes that must be
/// overloaded/specialized to create a new device adapter.
///
struct DeviceAdapterTag___ {  };
#endif //DAX_DOXYGEN_ONLY

namespace internal {

#ifdef DAX_DOXYGEN_ONLY

/// \brief Copy the contents of one ArrayHandle to another
///
/// Copies the contents of \c input to \c output. The array \c to will be
/// allocated to the appropriate size.
///
template<typename T, class CIn, class COut>
DAX_CONT_EXPORT void Copy(
    const dax::cont::ArrayHandle<T, CIn, DeviceAdapterTag___> &input,
    dax::cont::ArrayHandle<T, COut, DeviceAdapterTag___> &output,
    DeviceAdapterTag___);

/// \brief Compute an inclusive prefix sum operation on the input ArrayHandle.
///
/// Computes an inclusive prefix sum operation on the \c input ArrayHandle,
/// storing the results in the \c output ArrayHandle. InclusiveScan is similiar
/// to the stl partial sum function, exception that InclusiveScan doesn't do a
/// serial sumnation. This means that if you have defined a custom plus
/// operator for T it must be associative, or you will get inconsistent
/// results. When the input and output ArrayHandles are the same ArrayHandle
/// the operation will be done inplace.
///
template<typename T, class CIn, class COut>
DAX_CONT_EXPORT T InclusiveScan(
    const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag___> &input,
    dax::cont::ArrayHandle<T,COut,DeviceAdapterTag___>& output,
    DeviceAdapterTag___);

/// \brief Output is the first index in input for each item in values that wouldn't alter the ordering of input
///
/// LowerBounds is a vectorized search. From each value in \c values it finds
/// the first place the item can be inserted in the ordered \c input array and
/// stores the index in \c output.
///
/// \par Requirements:
/// \arg \c input must already be sorted
///
template<typename T, class CIn, class CVal, class COut>
DAX_CONT_EXPORT void LowerBounds(
    const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag___>& input,
    const dax::cont::ArrayHandle<T,CVal,DeviceAdapterTag___>& values,
    dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag___>& output,
    DeviceAdapterTag___);

/// \brief A special version of LowerBounds that does an in place operation.
///
/// This version of lower bounds performs an in place operation where each
/// value in the \c values_output array is replaced by the index in \c input
/// where it occurs. Because this is an in place operation, the of the arrays
/// is limited to dax::Id.
///
template<class CIn, class COut>
DAX_CONT_EXPORT void LowerBounds(
    const dax::cont::ArrayHandle<dax::Id,CIn,DeviceAdapterTag___>& input,
    dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag___>& values_output,
    DeviceAdapterTag___);

/// \brief Schedule many instances of a function to run on concurrent threads.
///
/// Calls the \c functor on several threads. This is the function used in the
/// control environment to spawn activity in the execution environment. \c
/// functor is a function-like object that can be invoked with the calling
/// specification <tt>functor(dax::Id index)</tt>. It also has a method called
/// from the control environment to establish the error reporting buffer with
/// the calling specification <tt>functor.SetErrorMessageBuffer(const
/// dax::exec::internal::ErrorMessageBuffer &errorMessage)</tt>. This object
/// can be stored in the functor's state such that if RaiseError is called on
/// it in the execution environment, an ErrorExecution will be thrown from
/// Schedule.
///
/// The argument of the invoked functor uniquely identifies the thread or
/// instance of the invocation. There should be one invocation for each index
/// in the range [0, \c numInstances].
///
template<class Functor>
DAX_CONT_EXPORT void Schedule(Functor functor,
                              dax::Id numInstances,
                              DeviceAdapterTag___);

/// \brief Unstable ascending sort of input array.
///
/// Sorts the contents of \c values so that they in ascending value. Doesn't
/// guarantee stability
///
template<typename T, class Container>
DAX_CONT_EXPORT void Sort(
    dax::cont::ArrayHandle<T,Container,DeviceAdapterTag___> &values,
    DeviceAdapterTag___);

/// \brief Performs stream compaction to remove unwanted elements in the input array. Output becomes the index values of input that are valid.
///
/// Calls the parallel primitive function of stream compaction on the \c
/// input to remove unwanted elements. The result of the stream compaction is
/// placed in \c output. The \c input values are used as the stream
/// compaction stencil while \c input indices are used as the values to place
/// into \c ouput. The size of \c output will be modified after this call as
/// we can't know the number of elements that will be removed by the stream
/// compaction algorithm.
///
template<typename T, class CIn, class COut>
DAX_CONT_EXPORT void StreamCompact(
    const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag___> &input,
    dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag___> &output,
    DeviceAdapterTag___);

/// \brief Performs stream compaction to remove unwanted elements in the input array.
///
/// Calls the parallel primitive function of stream compaction on the \c
/// input to remove unwanted elements. The result of the stream compaction is
/// placed in \c output. The values in \c stencil are used as the stream
/// compaction stencil while \c input values are placed into \c ouput. The
/// size of \c output will be modified after this call as we can't know the
/// number of elements that will be removed by the stream compaction
/// algorithm.
///
template<typename T, typename U, class Container>
static void StreamCompact(
    const dax::cont::ArrayHandle<T,Container,DeviceAdapterTag___> &input,
    const dax::cont::ArrayHandle<U,Container,DeviceAdapterTag___> &v,
    dax::cont::ArrayHandle<T,Container,DeviceAdapterTag___> &output,
    DeviceAdapterTag___);

/// \brief Reduce an array to only the unique values it contains
///
/// Removes all duplicate values in \c values that are adjacent to each
/// other. Which means you should sort the input array unless you want
/// duplicate values that aren't adjacent. Note the values array size might
/// be modified by this operation.
///
template<typename T, class Container>
DAX_CONT_EXPORT void Unique(
    dax::cont::ArrayHandle<T,Container,DeviceAdapterTag___>& values,
    DeviceAdapterTag___);

#endif //DAX_DOXYGEN_ONLY

/// \brief Class that manages data in the execution environment.
///
/// This templated class must be partially specialized for each
/// DeviceAdapterTag crated, which will define the implementation for that tag.
///
/// This is a class that is responsible for allocating data in the execution
/// environment and copying data back and forth between control and
/// execution. It is also expected that this class will automatically release
/// any resources in its destructor.
///
/// This class typically takes on one of two forms. If the control and
/// execution environments have seperate memory spaces, then this class
/// behaves how you would expect. It allocates/deallocates arrays and copies
/// data. However, if the control and execution environments share the same
/// memory space, this class should delegate all its operations to the
/// ArrayContainerControl. The latter can probably be implemented with a
/// trivial subclass of
/// dax::cont::internal::ArrayManagerExecutionShareWithControl.
///
template<typename T, class ArrayContainerControlTag, class DeviceAdapterTag>
class ArrayManagerExecution
#ifdef DAX_DOXYGEN_ONLY
{
private:
  typedef dax::cont::ArrayContainerControl<T,ArrayContainerControlTag>
      ContainerType;

public:
  /// The type of value held in the array (dax::Scalar, dax::Vector3, etc.)
  ///
  typedef T ValueType;

  /// An array portal that can be used in the execution environment to access
  /// portions of the arrays. This example defines the portal with a pointer,
  /// but any portal with methods that can be called and data that can be
  /// accessed from the execution environment can be used.
  ///
  typedef dax::exec::internal::ArrayPortalFromIterators<ValueType*> PortalType;

  /// Const version of PortalType.  You must be able to cast PortalType to
  /// PortalConstType.
  ///
  typedef dax::exec::internal::ArrayPortalFromIterators<const ValueType*>
      PortalConstType;

  /// Returns the number of values stored in the array.  Results are undefined
  /// if data has not been loaded or allocated.
  ///
  DAX_CONT_EXPORT dax::Id GetNumberOfValues() const;

  /// Allocates a large enough array in the execution environment and copies
  /// the given data to that array. The allocated array can later be accessed
  /// via the GetPortal method. If control and execution share arrays, then
  /// this method may save the iterators to be returned in the \c GetPortal*
  /// methods.
  ///
  DAX_CONT_EXPORT void LoadDataForInput(
      typename ContainerType::PortalType portal);

  /// Const version of LoadDataForInput. Functionally equivalent to the
  /// non-const version except that the non-const versions of GetPortal may not
  /// be available.
  ///
  DAX_CONT_EXPORT void LoadDataForInput(
      typename ContainerType::PortalConstType portal);

  /// Allocates an array in the execution environment of the specified size.
  /// If control and execution share arrays, then this class can allocate
  /// data using the given ArrayContainerExecution and remember its iterators
  /// so that it can be used directly in the exeuction environment.
  ///
  DAX_CONT_EXPORT void AllocateArrayForOutput(ContainerType &controlArray,
                                              dax::Id numberOfValues);

  /// Allocates data in the given ArrayContainerControl and copies data held
  /// in the execution environment (managed by this class) into the control
  /// array. If control and execution share arrays, this can be no operation.
  /// This method should only be called after AllocateArrayForOutput is
  /// called.
  ///
  DAX_CONT_EXPORT void RetrieveOutputData(ContainerType &controlArray) const;

  /// Similar to RetrieveOutputData except that instead of writing to the
  /// controlArray itself, it writes to the given control environment
  /// iterator. This allows the user to retrieve data without necessarily
  /// allocating an array in the ArrayContainerControl (assuming that control
  /// and exeuction have seperate memory spaces).
  ///
  template <class IteratorTypeControl>
  DAX_CONT_EXPORT void CopyInto(IteratorTypeControl dest) const;

  /// \brief Reduces the size of the array without changing its values.
  ///
  /// This method allows you to resize the array without reallocating it. The
  /// number of entries in the array is changed to \c numberOfValues. The data
  /// in the array (from indices 0 to \c numberOfValues - 1) are the same, but
  /// \c numberOfValues must be equal or less than the preexisting size
  /// (returned from GetNumberOfValues). That is, this method can only be used
  /// to shorten the array, not lengthen.
  ///
  DAX_CONT_EXPORT void Shrink(dax::Id numberOfValues);

  /// Returns an array portal that can be used in the execution environment.
  /// This portal was defined in either LoadDataForInput or
  /// AllocateArrayForOutput. If control and environment share memory space,
  /// this class may return the iterator from the \c controlArray.
  ///
  DAX_CONT_EXPORT PortalType GetPortal();

  /// Const version of GetPortal.
  ///
  DAX_CONT_EXPORT PortalConstType GetPortalConst() const;

  /// Frees any resources (i.e. memory) allocated for the exeuction
  /// environment, if any.
  ///
  DAX_CONT_EXPORT void ReleaseResources();
};
#else // DAX_DOXGEN_ONLY
;
#endif // DAX_DOXYGEN_ONLY

} // namespace internal

}
} // namespace dax::cont

// This is at the bottom of the file so that the templated class prototypes
// are declared before including the device adapter implementation.


//-----------------------------------------------------------------------------
#if DAX_DEVICE_ADAPTER == DAX_DEVICE_ADAPTER_SERIAL

#include <dax/cont/DeviceAdapterSerial.h>
#define DAX_DEFAULT_DEVICE_ADAPTER_TAG ::dax::cont::DeviceAdapterTagSerial

#elif DAX_DEVICE_ADAPTER == DAX_DEVICE_ADAPTER_CUDA

#include <dax/cuda/cont/DeviceAdapterCuda.h>
#define DAX_DEFAULT_DEVICE_ADAPTER_TAG ::dax::cuda::cont::DeviceAdapterTagCuda

#elif DAX_DEVICE_ADAPTER == DAX_DEVICE_ADAPTER_OPENMP

#include <dax/openmp/cont/DeviceAdapterOpenMP.h>
#define DAX_DEFAULT_DEVICE_ADAPTER_TAG ::dax::openmp::cont::DeviceAdapterTagOpenMP

#elif DAX_DEVICE_ADAPTER == DAX_DEVICE_ADAPTER_ERROR

#include <dax/cont/internal/DeviceAdapterError.h>
#define DAX_DEFAULT_DEVICE_ADAPTER_TAG ::dax::cont::internal::DeviceAdapterTagError

#elif (DAX_DEVICE_ADAPTER == DAX_DEVICE_ADAPTER_UNDEFINED) || !defined(DAX_DEVICE_ADAPTER)

#ifndef DAX_DEFAULT_DEVICE_ADAPTER_TAG
#warning If device adapter is undefined, DAX_DEFAULT_DEVICE_ADAPTER_TAG must be defined.
#endif

#endif


#endif //__dax_cont_DeviceAdapter_h
