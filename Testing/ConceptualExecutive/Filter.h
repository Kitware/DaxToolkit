#ifndef FILTER_H
#define FILTER_H

#include "DataSet.h"
#include "Executive.h"

#include <vector>

//------------------------------------------------------------------------------
template < typename Worklet >
class Filter
{
public:
  typedef typename Worklet::InputType InputType;
  typedef typename Worklet::OutputType OutputType;

  //we make other filters templated types friends
  //of this filter so we share the same Executive
  template <class OtherT> friend class Filter;

private:
  const Executive::ExecutivePtr Exec;
  const dax::HostArray<InputType>* InputArray;

public:
  Filter(const dax::HostArray<InputType> *t):
    Exec( new Executive() ),
    InputArray(t)
  {
    //this->Exec->connect<Worklet>(t, Worklet());
  }

  //constructor when connecting a filter to a different type of filter
  template<typename OtherT>
  Filter(const Filter<OtherT>& input):
    Exec(input.Exec),
    InputArray(NULL)
  {
    //this->Exec->connect< input::Worklet, Worklet >(input, Worklet());
  }

  //constructor when connecting a filter to an identical filter
  //needed as the T templated constructor doesn't work on those
  Filter(const Filter<Worklet>& input):
    Exec(input.Exec),
    InputArray(NULL)
  {
    //this->Exec->connect< input::Worklet, Worklet >(input, Worklet());
  }

  void run() const
  {
    this->Exec->run();
  }
};

#endif // FILTER_H
