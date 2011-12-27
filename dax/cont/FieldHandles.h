/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_FieldHandles_h
#define __dax_cont_FieldHandles_h

#include <string>
#include <dax/Types.h>
#include <dax/cont/internal/ArrayContainer.h>

namespace dax { namespace cont {

template<typename T>
class FieldHandlePoint
{
public:
  typedef T ValueType;

  FieldHandlePoint(const std::string &name);

  template<typename D>
  FieldHandlePoint(D& dataSet, const std::string &name);

  template<typename D>
  dax::Id fieldSize(const D& dataSet) const {return dataSet.numPoints();}

  //make the container that this handle has part
  //of the dataset g that is passed in
  template<typename D>
  void associate(D &dataSet);

  //set the execution array that container this handle
  //has. This is done so that when we associate to the dataset
  //we have actualy data.
  //Note: U should be some form of boost::shared_ptr as that
  //is what the array container is expecting
  template<typename U>
  void setExecutionArray(U array);

  //set the control array that container this handle
  //has. This is done so that when we associate to the dataset
  //we have actualy data.
  //Note: U should be some form of boost::shared_ptr as that
  //is what the array container is expecting
  template<typename U>
  void setControlArray(U array);

    template<typename U>
    boost::shared_ptr<U> arrayControl() const;

    template<typename U>
    boost::shared_ptr<U> arrayExecution() const;

private:
  dax::cont::internal::ArrayContainer<ValueType> Container;
  const std::string Name;
};

template<typename T>
class FieldHandleCell
{
public:
  typedef T ValueType;

  FieldHandleCell(const std::string &name);

  template<typename D>
  FieldHandleCell(D& dataSet, const std::string &name);

  template<typename D>
  dax::Id fieldSize(const D& dataSet) const {return dataSet.numCells();}

  //make the container that this handle has part
  //of the dataset g that is passed in
  template<typename D>
  void associate(D &dataSet);

  //set the execution array that container this handle
  //has. This is done so that when we associate to the dataset
  //we have actualy data.
  //Note: U should be some form of boost::shared_ptr as that
  //is what the array container is expecting
  template<typename U>
  void setExecutionArray(U array);

  //set the control array that container this handle
  //has. This is done so that when we associate to the dataset
  //we have actualy data.
  //Note: U should be some form of boost::shared_ptr as that
  //is what the array container is expecting
  template<typename U>
  void setControlArray(U array);

  template<typename U>
  boost::shared_ptr<U> arrayControl() const;

  template<typename U>
  boost::shared_ptr<U> arrayExecution() const;

private:
  dax::cont::internal::ArrayContainer<ValueType> Container;
  const std::string Name;
};

//------------------------------------------------------------------------------
template<typename T>
FieldHandlePoint<T>::FieldHandlePoint(const std::string &name):
  Name(name)
{
}

//------------------------------------------------------------------------------
template<typename T> template<typename D>
FieldHandlePoint<T>::FieldHandlePoint(D& dataSet, const std::string &name):
  Name(name),
  Container(dataSet.fieldsPoint().get(T(),name))
{
  //populate Container from the dataset
}

//------------------------------------------------------------------------------
template<typename T> template<typename D>
void FieldHandlePoint<T>::associate(D &dataSet)
{
  dataSet.fieldsPoint().addArray(this->Name,this->Container);
}

//------------------------------------------------------------------------------
template<typename T> template<typename U>
inline void FieldHandlePoint<T>::setExecutionArray(U array)
{
  this->Container.setArrayExecution(array);
}

//------------------------------------------------------------------------------
template<typename T> template<typename U>
inline void FieldHandlePoint<T>::setControlArray(U array)
{
  this->Container.setArrayControl(array);
}

//------------------------------------------------------------------------------
template<typename T> template<typename U>
inline boost::shared_ptr<U> FieldHandlePoint<T>::arrayControl() const
{
  //these have the same call signature and name as the container
  //we call so that the same function in DeviceArray can be used for both
  return this->Container.template arrayControl<U>();
}

//------------------------------------------------------------------------------
template<typename T> template<typename U>
inline boost::shared_ptr<U> FieldHandlePoint<T>::arrayExecution() const
{
  //these have the same call signature and name as the container
  //we call so that the same function in DeviceArray can be used for both
  return this->Container.template arrayExecution<U>();
}



//------------------------------------------------------------------------------
template<typename T>
FieldHandleCell<T>::FieldHandleCell(const std::string &name):
  Name(name)
{
}

//------------------------------------------------------------------------------
template<typename T> template<typename D>
FieldHandleCell<T>::FieldHandleCell(D& dataSet, const std::string &name):
  Name(name),
  Container(dataSet.fieldsCell().get(T(),name))
{
  //populate Container from the dataset
}

//------------------------------------------------------------------------------
template<typename T> template<typename D>
void FieldHandleCell<T>::associate(D &dataSet)
{
  dataSet.fieldsCell().addArray(this->Name,this->Container);
}

//------------------------------------------------------------------------------
template<typename T> template<typename U>
inline void FieldHandleCell<T>::setExecutionArray(U array)
{
  this->Container.setArrayExecution(array);
}

//------------------------------------------------------------------------------
template<typename T> template<typename U>
inline void FieldHandleCell<T>::setControlArray(U array)
{
  this->Container.setArrayControl(array);
}

//------------------------------------------------------------------------------
template<typename T> template<typename U>
inline boost::shared_ptr<U> FieldHandleCell<T>::arrayControl() const
{
  //these have the same call signature and name as the container
  //we call so that the same function in DeviceArray can be used for both
  return this->Container.template arrayControl<U>();
}

//------------------------------------------------------------------------------
template<typename T> template<typename U>
inline boost::shared_ptr<U> FieldHandleCell<T>::arrayExecution() const
{
  //these have the same call signature and name as the container
  //we call so that the same function in DeviceArray can be used for both
  return this->Container.template arrayExecution<U>();
}

} }


#endif // __dax_cont_FieldHandles_h
