/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cuda_cont_Model_h
#define __dax_cuda_cont_Model_h

namespace dax { namespace cuda { namespace cont {

template< typename DataType>
class Model
{
public:
  Model(DataType input):
    Data(input)
    {
    }

  DataType Data;
};

} } }

#endif
