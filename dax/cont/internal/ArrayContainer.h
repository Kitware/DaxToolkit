#ifndef dax_cont_internal_ArrayContainer_h
#define dax_cont_internal_ArrayContainer_h

#include <boost/shared_ptr.hpp>

namespace dax { namespace cont { namespace internal {

// Holds a shared_pointer for the array in Control and Excution enviornment
// It currently only manages ownership and locality it doesn't
// try to keep the contents identical between the arrays.
template<typename Type>
class ArrayContainer
{
public:
  typedef Type ValueType;

  ArrayContainer(){}

  bool hasControlArray() const
    {
    return (this->ControlArray != NULL);
    }

  bool hasExecutionArray() const
    {
    return (this->ExecutionArray != NULL);
    }

  template<typename T>
  void setArrayControl(boost::shared_ptr<T> array)
    {
    this->ControlArray = array;
    }

  template<typename T>
  void setArrayExecution(boost::shared_ptr<T> array)
    {
    this->ExecutionArray = array;
    }

  template<typename T>
  boost::shared_ptr<T> arrayControl()
    {
    if(this->ControlArray)
      {
      return boost::static_pointer_cast<T>(this->ControlArray);
      }

    boost::shared_ptr<T> controlArray(new T());
    if(this->ExecutionArray)
      {
      //copy the array from control to execution
      //this is a trick to get around the type erasure on control array
      controlArray = controlArray->convert(this->ExecutionArray);
      }
    this->setArrayControl(controlArray);
    return boost::static_pointer_cast<T>(this->ControlArray);
    }

  template<typename T>
  boost::shared_ptr<T> arrayExecution( )
    {
    if(this->ExecutionArray)
      {
      return boost::static_pointer_cast<T>(this->ExecutionArray);
      }

    boost::shared_ptr<T> executionArray(new T());
    if(this->ControlArray)
      {
      //copy the array from control to execution
      //this is a trick to get around the type erasure on control array
      executionArray = executionArray->convert(this->ControlArray);
      }
    this->setArrayExecution(executionArray);
    return boost::static_pointer_cast<T>(this->ExecutionArray);
    }

private:
  ArrayContainer& operator=(const ArrayContainer& rhs);

  //see the documentation of shared pointer that it can be
  //used to hold arbitrary datra
  //the issue is that is type erasure so assignment is a real big pain
  boost::shared_ptr<void> ControlArray;
  boost::shared_ptr<void> ExecutionArray;

};

} } }

#endif dax_cont_internal_ArrayContainer_h
