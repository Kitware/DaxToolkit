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
#ifndef __dax_cont_internal_DeviceAdapterAlgorithm_h
#define __dax_cont_internal_DeviceAdapterAlgorithm_h

#include <dax/cont/internal/ArrayManagerExecution.h>
#include <dax/cont/internal/DeviceAdapterTag.h>

#if DAX_DEVICE_ADAPTER == DAX_DEVICE_ADAPTER_SERIAL
#include <dax/cont/internal/DeviceAdapterAlgorithmSerial.h>
#elif DAX_DEVICE_ADAPTER == DAX_DEVICE_ADAPTER_CUDA
#include <dax/cuda/cont/internal/DeviceAdapterAlgorithmCuda.h>
#elif DAX_DEVICE_ADAPTER == DAX_DEVICE_ADAPTER_OPENMP
#include <dax/openmp/cont/internal/DeviceAdapterAlgorithmOpenMP.h>
#endif

namespace dax {
namespace cont {
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

/// \brief Compute an exclusive prefix sum operation on the input ArrayHandle.
///
/// Computes an exclusive prefix sum operation on the \c input ArrayHandle,
/// storing the results in the \c output ArrayHandle. ExclusiveScan is similiar
/// to the stl partial sum function, exception that ExclusiveScan doesn't do a
/// serial sumnation. This means that if you have defined a custom plus
/// operator for T it must be associative, or you will get inconsistent
/// results. When the input and output ArrayHandles are the same ArrayHandle
/// the operation will be done inplace.
///
template<typename T, class CIn, class COut>
DAX_CONT_EXPORT T ExclusiveScan(
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

/// \brief Output is the first index in input for each item in values that wouldn't alter the ordering of input
///
/// UpperBounds is a vectorized search. From each value in \c values it finds
/// the first place the item can be inserted in the ordered \c input array and
/// stores the index in \c output.
///
/// \par Requirements:
/// \arg \c input must already be sorted
///
template<typename T, class CIn, class CVal, class COut>
DAX_CONT_EXPORT void UpperBounds(
    const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag___>& input,
    const dax::cont::ArrayHandle<T,CVal,DeviceAdapterTag___>& values,
    dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag___>& output,
    DeviceAdapterTag___);

/// \brief A special version of UpperBounds that does an in place operation.
///
/// This version of lower bounds performs an in place operation where each
/// value in the \c values_output array is replaced by the index in \c input
/// where it occurs. Because this is an in place operation, the of the arrays
/// is limited to dax::Id.
///
template<class CIn, class COut>
DAX_CONT_EXPORT void UpperBounds(
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

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_DeviceAdapterAlgorithm_h
