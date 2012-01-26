/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_thrust_cont_StreamCompact_h
#define __dax_thrust_cont_StreamCompact_h

#include <dax/Types.h>
#include <dax/Functional.h>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/scan.h>



namespace dax {
namespace thrust {
namespace cont {


template<typename InputIterator,
         typename StencilVector,
         typename OutputVector,
         typename Predicate>
DAX_EXEC_CONT_EXPORT void RemoveIf(InputIterator valuesFirst,
                                    InputIterator valuesEnd,
                                    const StencilVector& stencil,
                                    OutputVector& output,
                                    Predicate pred)
{
  //we need to do some profiling on what way of doing stream compaction
  //is the fastest. PISTON uses an inclusive scan and then upper_bound.
  //first get the correct size for output. What about the speed
  //between remove_copy_if and copy_if

  //first get the correct size for output
  int numLeft = ::thrust::reduce(stencil.begin(),stencil.end());
  output.resize(numLeft);

  //remove any item that matches the predicate
  ::thrust::copy_if(valuesFirst,
                     valuesEnd,
                     stencil.begin(),
                     output.begin(),
                     pred);
}


template<typename T, typename U>
DAX_EXEC_CONT_EXPORT void streamCompact(const ::thrust::device_vector<T>& input,
                          const ::thrust::device_vector<U>& stencil,
                          ::thrust::device_vector<T>& output)
{
  //do the copy step, remember the input is the stencil
  //set the predicate to be the identity of type U so we get rid of anything
  //that matches the default constructor
  dax::thrust::cont::RemoveIf(input.begin(),
                            input.end(),
                            stencil,
                            output,
                            dax::not_identity<U>());
}

template<typename T>
DAX_EXEC_CONT_EXPORT void streamCompact(
    const ::thrust::device_vector<T>& input,
    ::thrust::device_vector<T>& output)
{
  //do the copy step, remember the input is the stencil
  //set the predicate to be the identity of type T so we get rid of anything
  //that matches the default constructor
  dax::thrust::cont::RemoveIf(::thrust::make_counting_iterator<T>(0),
                              ::thrust::make_counting_iterator<T>(input.size()),
                              input,
                              output,
                              dax::not_identity<T>());
}

}
}
} // namespace dax::thrust::cont

#endif //__dax_thrust_cont_StreamCompact_h
