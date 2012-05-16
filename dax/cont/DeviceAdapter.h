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

#include <dax/internal/ExportMacros.h>

#ifndef DAX_DEFAULT_DEVICE_ADAPTER
#ifdef DAX_CUDA
#include <dax/cuda/cont/DeviceAdapterCuda.h>
#else // DAX_CUDA
#ifdef DAX_OPENMP
#include <dax/openmp/cont/DeviceAdapterOpenMP.h>
#else // DAX_OPENMP
#include <dax/cont/DeviceAdapterSerial.h>
#endif // DAX_OPENMP
#endif // DAX_CUDA
#endif // DAX_DEFAULT_DEVICE_ADAPTER

namespace dax {
namespace cont {

#ifdef DAX_DOXYGEN_ONLY

/// \brief Interface between generic Dax classes and parallel devices.
///
/// A DeviceAdapter is a class with several classes static methods that provide
/// mechanisms to run algorithms on a type of parallel device. The class
/// dax::cont::DeviceAdapter does not actually exist. Rather, this
/// documentation is provided to describe the interface for a DeviceAdapter.
/// Loading the dax/cont/DeviceAdapter.h header file will set a default device
/// adapter appropriate for the current compile environment. The default
/// adapter can be overloaded by including the header file for a different
/// adapter (for example, DeviceAdapterSerial.h). This overloading should be
/// done \em before loading in any other Dax header files. Failing to do so
/// could create inconsistencies in the default adapter used among classes.
///
class DeviceAdapter
{
public:
  /// \brief Copy the contents of one ArrayHandle to another
  ///
  /// Copies the contents of \c from to \c to. The array \c to must be of the same
  /// type as \c from and be at least the same size as \c from. If \c to is larger
  /// that \c from the excess range will have undefined values.
  /// Requirements: \c from must already be allocated in the execution environment
  ///
  template<typename T, template <typename> class Container>
  static void Copy(
      const dax::cont::ArrayHandle<T,Container,DeviceAdapter>& from,
      dax::cont::ArrayHandle<T,Container,DeviceAdapter>& to);

  /// \brief Compute an inclusive prefix sum operation on the input ArrayHandle.
  ///
  /// Computes an inclusive prefix sum operation on the \c input ArrayHandle, storing
  /// the results in the \c output ArrayHandle. InclusiveScan is similiar to the
  /// stl partial sum function, exception that InclusiveScan doesn't do a serial
  /// sumnation. This means that if you have defined a custom plus operator for
  /// T it must be associative, or you will get inconsistent results.
  /// When the input and output ArrayHandles are the same ArrayHandle the operation
  /// will be done inplace.
  /// \par Requirements:
  /// \arg \c input must already be allocated in the execution environment
  /// \arg \c input and \c output must be the same size
  ///
  template<typename T, template <typename> class Container>
  static T InclusiveScan(
      const dax::cont::ArrayHandle<T,Container,DeviceAdapter> &input,
      dax::cont::ArrayHandle<T,Container,DeviceAdapter>& output);

  /// \brief Output is the first index in input for each item in values that wouldn't alter the ordering of input
  ///
  /// LowerBounds is a vectorized search. From each value in \c values it finds
  /// the first place the item can be inserted in the ordered \c input array
  /// and stores the index in \c output.
  ///
  /// \note \c values and \c output can be the same array.
  ///
  /// \par Requirements:
  /// \arg \c input must already be sorted
  /// \arg \c input must already be allocated in the execution environment
  /// \arg \c values must already be allocated in the execution environment
  /// \arg \c values and \c output must be the same size
  template<typename T, template <typename> class Container>
  static void LowerBounds(
      const dax::cont::ArrayHandle<T,Container,DeviceAdapter>& input,
      const dax::cont::ArrayHandle<T,Container,DeviceAdapter>& values,
      dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapter>& output);

  /// \brief Schedule many instances of a function to run on concurrent threads.
  ///
  /// Calls the \c functor on several threads. This is the function used in the
  /// control environment to spawn activity in the execution environment. \c
  /// functor is a function-like object that can be invoked with the calling
  /// specification <tt>functor(Parameters parameters, dax::Id index,
  /// DeviceAdapter::ExectionAdapter::ErrorHandler errorHandler)</tt>. The
  /// first argument is the \c parameters passed through. The second argument
  /// uniquely identifies the thread or instance of the invocation. There
  /// should be one invocation for each index in the range [0, \c
  /// numInstances]. The third argment contains an ErrorHandler that can be
  /// used to raise an error in the functor.  The final argument is not used
  /// for anything but indirectly specifying the template paramters.
  ///
  template<class Functor,
           class Parameters,
           template <typename> class ArrayContainerControl>
  static void Schedule(Functor functor,
                       Parameters parameters,
                       dax::Id numInstances,
                       ExecutionAdapter<ArrayContainerControl>);

  /// \brief Unstable ascending sort of input array.
  ///
  /// Sorts the contents of \c values so that they in ascending value. Doesn't
  /// guarantee stability
  ///
  /// \par Requirements;
  /// \arg \c values must already be allocated in the execution environment
  ///
  template<typename T, template <typename> class Container>
  static void Sort(dax::cont::ArrayHandle<T,Container,DeviceAdapter>& values);

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
  /// \par Requirements:
  /// \arg \c input must already be allocated in the execution environment
  ///
  template<typename T, template <typename> class Container>
  static void StreamCompact(
      const dax::cont::ArrayHandle<T,Container,DeviceAdapter> &input,
      dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapter> &output);

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
  /// \par Requirements:
  /// \arg \c input must already be allocated in the execution environment
  /// \arg \c stencil must already be allocated in the execution environment
  ///
  template<typename T, typename U>
  static void StreamCompact(
      const dax::cont::ArrayHandle<T,Container,DeviceAdapter> &input,
      const dax::cont::ArrayHandle<U,Container,DeviceAdapter> &v,
      dax::cont::ArrayHandle<T,Container,DeviceAdapter> &output);

  /// \brief Reduce an array to only the unique values it contains
  ///
  /// Removes all duplicate values in \c values that are adjacent to each
  /// other. Which means you should sort the input array unless you want
  /// duplicate values that aren't adjacent. Note the values array size might
  /// be modified by this operation.
  ///
  /// \par Requirements:
  /// \arg \c values must already be allocated in the execution environment
  ///
  template<typename T, template <typename> class Container>
  static void Unique(dax::cont::ArrayHandle<T,Container,DeviceAdapter>& values);

  /// \brief Class that manages data in the execution environment.
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
  template<typename T, template <typename> class ArrayContainerControl>
  class ArrayManagerExecution
  {
  public:
    /// The type of value held in the array (dax::Scalar, dax::Vector3, etc.)
    ///
    typedef T ValueType;

    /// An iterator that can be used in the execution environment to access
    /// portions of the arrays. This example defines the iterator as a pointer,
    /// but any random access iterator is valid.
    ///
    typedef ValueType *IteratorType;

    /// Const version of IteratorType.
    ///
    typedef const ValueType *IteratorConstType;

    /// Allocates a large enough array in the execution environment and copies
    /// the given data to that array. The allocated array can later be accessed
    /// via the GetIteratorBegin method. If control and execution share arrays,
    /// then this method may save the iterators to be returned in the \c
    /// GetIterator* methods.
    ///
    void LoadDataForInput(
        typename ArrayContainerControl<ValueType>::IteratorType beginIterator,
        typename ArrayContainerControl<ValueType>::IteartorType endIterator);

    /// Const version of LoadDataForInput.  Functionally equivalent to the
    /// non-const version except that the non-const versions of GetIterator*
    /// may not be available.
    void LoadDataForInput(
        typename ArrayContainerControl<ValueType>::IteratorConstType beginIterator,
        typename ArrayContainerControl<ValueType>::IteartorConstType endIterator);

    /// Allocates an array in the execution environment of the specified size.
    /// If control and execution share arrays, then this class can allocate
    /// data using the given ArrayContainerExecution and remember its iterators
    /// so that it can be used directly in the exeuction environment.
    ///
    void AllocateArrayForOutput(ArrayContainerControl<ValueType> &controlArray,
                                dax::Id numberOfValues);

    /// Allocates data in the given ArrayContainerControl and copies data held
    /// in the execution environment (managed by this class) into the control
    /// array. If control and execution share arrays, this can be no operation.
    /// This method should only be called after AllocateArrayForOutput is
    /// called.
    ///
    void RetrieveOutputData(
        ArrayContainerControl<ValueType> &controlArray) const;

    /// Similar to RetrieveOutputData except that instead of writing to the
    /// controlArray itself, it writes to the given control environment
    /// iterator. This allows the user to retrieve data without necessarily
    /// allocating an array in the ArrayContainerControl (assuming that control
    /// and exeuction have seperate memory spaces).
    ///
    template <class IteratorTypeControl>
    void CopyInto(IteratorTypeControl dest) const;

    /// \brief Reduces the size of the array without changing its values.
    ///
    /// This method allows you to resize the array without reallocating it. The
    /// number of entries in the array is changed to \c numberOfValues. The data
    /// in the array (from indices 0 to \c numberOfValues - 1) are the same, but
    /// \c numberOfValues must be equal or less than the preexisting size
    /// (returned from GetNumberOfValues). That is, this method can only be used
    /// to shorten the array, not lengthen.
    ///
    void Shrink(dax::Id numberOfValues);

    /// Returns an iterator that can be used in the execution environment. This
    /// iterator was defined in either LoadDataForInput or
    /// AllocateArrayForOutput. If control and environment share memory space,
    /// this class may return the iterator from the \c controlArray.
    ///
    IteratorType GetIteratorBegin();

    /// Returns an iterator that can be used in the execution environment. This
    /// iterator was defined in either LoadDataForInput or
    /// AllocateArrayForOutput. If control and environment share memory space,
    /// this class may return the iterator from the \c controlArray.
    ///
    IteratorType GetIteratorEnd();

    /// Const version of GetIteratorBegin.
    ///
    IteratorConstType GetIteratorConstBegin() const;

    /// Const version of GetIteratorEnd.
    ///
    IteratorConstType GetIteratorConstEnd() const;

    /// Frees any resources (i.e. memory) allocated for the exeuction
    /// environment, if any.
    ///
    void ReleaseResources();
  };

  /// \brief Adapter for execution environment.
  ///
  /// Classes in the execution environment use this class in their template
  /// arguments. This class allows the execution environment to adapt to
  /// different devices.
  ///
  template<template <typename> class ArrayContainerControl>
  class ExecutionAdapter
  {
  public:
    /// This structure contains iterators that can be used to access the arrays
    /// representing fields.  The funny templating of the structure containing
    /// iterators is to handle the case of iterators that are pointers, which
    /// cannot be partially templated (at least before C++11, which is not yet
    /// widely adopted).
    ///
    template <typename T>
    struct FieldStructures
    {
      typedef typename DeviceAdapter::ArrayManagerExecution<
          T,ArrayContainerControl>::IteratorType IteratorType;
      typedef typename DeviceAdapter::ArrayManagerExecution<
          T,ArrayContainerControl>::IteratorConstType IteratorConstType;
    };

    /// This class is constructed in work objects so that they can raise
    /// errors. The work object will simply call the RaiseError method. This
    /// method should either set the state of the object to signal the error or
    /// throw an exception.
    ///
    class ErrorHandler
    {
    public:
      void RaiseError(const char *message) const;
    };
  };
};

#endif //DAX_DOXYGEN_ONLY

}
}

//forward declare all the different device adapters so that
//ArrayHandle can be use them as a friend class
namespace dax { namespace cont { struct DeviceAdapterSerial; }}
namespace dax { namespace cuda { namespace cont { struct DeviceAdapterCuda; }}}
namespace dax { namespace openmp {  namespace cont { struct DeviceAdapterOpenMP; }}}

#endif //__dax_cont_DeviceAdapter_h
