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
  template<typename T>
  static void Copy(const dax::cont::ArrayHandle<T,DeviceAdapter>& from,
                   dax::cont::ArrayHandle<T,DeviceAdapter>& to);

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
  template<typename T>
  static T InclusiveScan(const dax::cont::ArrayHandle<T,DeviceAdapter> &input,
                         dax::cont::ArrayHandle<T,DeviceAdapter>& output);

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
  template<typename T>
  static void LowerBounds(
      const dax::cont::ArrayHandle<T,DeviceAdapter>& input,
      const dax::cont::ArrayHandle<T,DeviceAdapter>& values,
      dax::cont::ArrayHandle<dax::Id,DeviceAdapter>& output);

  /// \brief Schedule many instances of a function to run on concurrent threads.
  ///
  /// Calls the \c functor on several threads. This is the function used in the
  /// control environment to spawn activity in the execution environment. \c
  /// functor is a function-like object that can be invoked with the calling
  /// specification <tt>functor(Parameters parameters, dax::Id index,
  /// dax::exec::internal::ErrorHandler errorHandler)</tt>. The first argument
  /// is the \c parameters passed through. The second argument uniquely
  /// identifies the thread or instance of the invocation. There should be one
  /// invocation for each index in the range [0, \c numInstances]. The third
  /// argment contains an ErrorHandler that can be used to raise an error in
  /// the functor.
  ///
  template<class Functor, class Parameters>
  static void Schedule(Functor functor,
                       Parameters parameters,
                       dax::Id numInstances);

  /// \brief Unstable ascending sort of input array.
  ///
  /// Sorts the contents of \c values so that they in ascending value. Doesn't
  /// guarantee stability
  ///
  /// \par Requirements;
  /// \arg \c values must already be allocated in the execution environment
  ///
  template<typename T>
  static void Sort(dax::cont::ArrayHandle<T,DeviceAdapter>& values);

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
  template<typename T>
  static void StreamCompact(
      const dax::cont::ArrayHandle<T,DeviceAdapter> &input,
      dax::cont::ArrayHandle<dax::Id,DeviceAdapter> &output);

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
      const dax::cont::ArrayHandle<T,DeviceAdapter> &input,
      const dax::cont::ArrayHandle<U,DeviceAdapter> &v,
      dax::cont::ArrayHandle<T,DeviceAdapter> &output);

  /// \brief Reduce an array to only the unique values it contains
  ///
  /// Removes all duplicate values in \c values which are adjacent to each
  /// other. Which means you should sort the input array unless you want
  /// duplicate values which aren't adjacent. Note the values array size might
  /// be modified by this operation.
  ///
  /// \par Requirements:
  /// \arg \c values must already be allocated in the execution environment
  ///
  template<typename T>
  static void Unique(dax::cont::ArrayHandle<T,DeviceAdapter>& values);

  /// \brief Class that manages data in the execution environment.
  ///
  /// This is a class that is responsible for allocating data in the execution
  /// environment and copying data back and forth between control and execution
  /// environment. It is also expected that this class will automatically
  /// release any resources in its destructor.
  ///
  template<class T>
  class ArrayContainerExecution
  {
  public:

    /// Allocates an array on the device large enough to hold the given number
    /// of entries.
    ///
    void Allocate(dax::Id numEntries);

    /// Copies the data pointed to by the passed in \c iterators (assumed to be
    /// in the control environment), into the array in the execution
    /// environment managed by this class.
    ///
    template<class Iterator>
    void CopyFromControlToExecution(
        const dax::cont::internal::IteratorContainer<Iterator> &iterators);

    /// Copies the data from the array in the execution environment managed by
    /// this class into the memory passed in the \c iterators (assumed to be in
    /// the control environment).
    ///
    template<class Iterator>
    void CopyFromExecutionToControl(
        const dax::cont::internal::IteratorContainer<Iterator> &iterators);

    /// Frees any resources (i.e. memory) on the device.
    ///
    void ReleaseResources();

    /// Gets a DataArray that is valid in the execution environment.
    ///
    dax::internal::DataArray<T> GetExecutionArray();
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
