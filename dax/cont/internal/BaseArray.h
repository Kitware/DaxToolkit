#ifndef __dax_cont_BaseArray_h
#define __dax_cont_BaseArray_h

#include <dax/Types.h>
namespace dax { namespace cont { namespace internal {

class BaseArray
{
public:
  virtual ~BaseArray(){}
  virtual std::string name() const=0;
  };

} } }

#endif // __dax_cont_BaseArray_h
