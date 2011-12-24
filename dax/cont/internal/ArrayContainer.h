/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef dax_cont_internal_ArrayContainer_h
#define dax_cont_internal_ArrayContainer_h

#include "Object.h"
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

  bool hasControlArray() const;
  bool hasExecutionArray() const;

  template<typename T>
  void setArrayControl(boost::shared_ptr<T> array);

  template<typename T>
  void setArrayExecution(boost::shared_ptr<T> array);

  template<typename T>
  boost::shared_ptr<T> arrayControl() const;

  template<typename T>
  boost::shared_ptr<T> arrayExecution( ) const;

private:
  //see the documentation of shared pointer that it can be
  //used to hold arbitrary datra
  //the issue is that is type erasure so assignment is a real big pain
  mutable boost::shared_ptr<void> ControlArray;
  mutable boost::shared_ptr<void> ExecutionArray;

};

//------------------------------------------------------------------------------
template<typename Type>
inline bool ArrayContainer<Type>::hasControlArray() const
{
  return (this->ControlArray != NULL);
}

//------------------------------------------------------------------------------
template<typename Type>
inline bool ArrayContainer<Type>::hasExecutionArray() const
{
return (this->ExecutionArray != NULL);
}

//------------------------------------------------------------------------------
template<typename Type> template<typename T>
inline void ArrayContainer<Type>::setArrayControl(boost::shared_ptr<T> array)
{
  std::cout << "setArrayControl" << std::endl;
  this->ControlArray = array;
}

//------------------------------------------------------------------------------
template<typename Type> template<typename T>
inline void ArrayContainer<Type>::setArrayExecution(boost::shared_ptr<T> array)
{
  this->ExecutionArray = array;
}

//------------------------------------------------------------------------------
template<typename Type> template<typename T>
boost::shared_ptr<T> ArrayContainer<Type>::arrayControl() const
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
  this->ControlArray = controlArray;
  return boost::static_pointer_cast<T>(this->ControlArray);
}

//------------------------------------------------------------------------------
template<typename Type> template<typename T>
boost::shared_ptr<T> ArrayContainer<Type>::arrayExecution( ) const
{
  if(this->ExecutionArray)
    {
    return boost::static_pointer_cast<T>(this->ExecutionArray);
    }
  std::cout << "creating a new execution array!" <<std::endl;
  boost::shared_ptr<T> executionArray(new T());
  if(this->ControlArray)
    {
    std::cout << "assigning new array from control!" <<std::endl;
    //copy the array from control to execution
    //this is a trick to get around the type erasure on control array
    executionArray = executionArray->convert(this->ControlArray);
    }
  this->ExecutionArray = executionArray;
  return boost::static_pointer_cast<T>(this->ExecutionArray);
}

} } }

#endif dax_cont_internal_ArrayContainer_h
