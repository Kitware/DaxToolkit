/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_thrust_cont_StreamCompact_h
#define __dax_thrust_cont_StreamCompact_h

#include <dax/Types.h>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>


namespace dax {
namespace thrust {
namespace cont {


template<typename T>
DAX_CONT_EXPORT static void streamCompact(const ::thrust::device_vector<T>& input,
                          const ::thrust::device_vector<dax::Id>& stencil,
                          ::thrust::device_vector<T>& output)
{
  ::thrust::device_vector<dax::Id> temp(input.size());
  //use the stencil to find the new index value for each item
  ::thrust::inclusive_scan(stencil.begin(), stencil.end(),
                         temp.begin());
  int numLeft = temp.back();
  output.resize(numLeft);

  // generate the value for each item in the scatter
  ::thrust::upper_bound(temp.begin(), temp.end(),
                        input.begin(),input.end(),
                        output.begin());
}


DAX_CONT_EXPORT static void streamCompact(
    const ::thrust::device_vector<dax::Id>& input,
    ::thrust::device_vector<dax::Id>& output)
{
  ::thrust::device_vector<dax::Id> temp(input.size());
  //use the values in input to find the new index value for each item
  ::thrust::inclusive_scan(input.begin(), input.end(),
                         temp.begin());
  int numLeft = temp.back();
  output.resize(numLeft);

  // generate the value for each item in the scatter
  ::thrust::upper_bound(temp.begin(), temp.end(),
                        ::thrust::make_counting_iterator<dax::Id>(0),
                        ::thrust::make_counting_iterator<dax::Id>(numLeft),
                        output.begin());
}

}
}
} // namespace dax::thrust::cont

#endif //__dax_thrust_cont_StreamCompact_h
