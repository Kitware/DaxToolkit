/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_internal_IteratorContainer_h
#define __dax_cont_internal_IteratorContainer_h

#include <dax/Types.h>

#include <iterator>

namespace dax {
namespace cont {
namespace internal {

template<typename IterT>
class IteratorContainer
{
public:
  typedef IterT IteratorType;
  typedef typename std::iterator_traits<IteratorType>::value_type ValueType;

  IteratorContainer() : Valid(false) { }
  IteratorContainer(IteratorType begin, IteratorType end)
    : BeginIterator(begin), EndIterator(end), Valid(true) { }

  bool IsValid() const { return this->Valid; }
  void Invalidate() { this->Valid = false; }

  IteratorType GetBeginIterator() const { return this->BeginIterator; }
  IteratorType GetEndIterator() const { return this->EndIterator; }

  dax::Id GetNumberOfEntries() const {
    return this->GetEndIterator() - this->GetBeginIterator();
  }

private:
  IteratorContainer(const IteratorContainer &); // Not implemented
  void operator=(const IteratorContainer &);    // Not implemented

  IteratorType BeginIterator;
  IteratorType EndIterator;
  bool Valid;
};

}
}
}

#endif //__dax_cont_internal_IteratorContainer_h
