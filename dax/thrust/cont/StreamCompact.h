/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_thrust_cont_StreamCompact_h
#define __dax_thrust_cont_StreamCompact_h

#include <dax/Types.h>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/logical.h>
#include <thrust/scan.h>



namespace dax {
namespace thrust {
namespace cont {


template<typename InputIterator,
         typename StencilVector,
         typename OutputVector,
         typename Predicate>
DAX_CONT_EXPORT static void CopyIf(InputIterator valuesFirst,
                                    InputIterator valuesEnd,
                                    const StencilVector& stencil,
                                    OutputVector& output,
                                    Predicate pred)
{
  //we need to do some profiling on what way of doing stream compaction
  //is the fastest. PISTON uses an inclusive scan and then upper_bound.
  //first get the correct size for output

  //first get the correct size for output
  int numLeft = ::thrust::reduce(stencil.begin(),stencil.end());
  output.resize(numLeft);

  ::thrust::copy_if(valuesFirst,
                    valuesEnd,
                    stencil.begin(),
                    output.begin(),
                    pred);
}


template<typename T>
DAX_CONT_EXPORT static void streamCompact(const ::thrust::device_vector<T>& input,
                          const ::thrust::device_vector<dax::Id>& stencil,
                          ::thrust::device_vector<T>& output)
{
  //do the copy step, remember the input is the stencil
  dax::thrust::cont::CopyIf(input.begin(),
                            input.end(),
                            stencil,
                            output,
                            ::thrust::identity<dax::Id>());
}


DAX_CONT_EXPORT static void streamCompact(
    const ::thrust::device_vector<dax::Id>& input,
    ::thrust::device_vector<dax::Id>& output)
{
  //do the copy step, remember the input is the stencil
  dax::thrust::cont::CopyIf(::thrust::make_counting_iterator<dax::Id>(0),
                            ::thrust::make_counting_iterator<dax::Id>(input.size()),
                            input,
                            output,
                            ::thrust::identity<dax::Id>());
}

}
}
} // namespace dax::thrust::cont

#endif //__dax_thrust_cont_StreamCompact_h
