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
class pointFieldHandle
{
public:
  typedef T ValueType;

  pointFieldHandle(const std::string &name);

  template<typename G>
  dax::Id fieldSize(const G& g) const {return g.numPoints();}

  //make the container that this handle has part
  //of the dataset g that is passed in
  template<typename G>
  void associate(G &g);

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

private:
  dax::cont::internal::ArrayContainer<ValueType> Container;
  const std::string Name;
};

template<typename T>
class cellFieldHandle
{
public:
  typedef T ValueType;

  cellFieldHandle(const std::string &name);

  template<typename G>
  dax::Id fieldSize(const G& g) const {return g.numCells();}

  //make the container that this handle has part
  //of the dataset g that is passed in
  template<typename G>
  void associate(G &g);

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

private:
  dax::cont::internal::ArrayContainer<ValueType> Container;
  const std::string Name;
};

//------------------------------------------------------------------------------
template<typename T>
pointFieldHandle<T>::pointFieldHandle(const std::string &name):
  Name(name)
{
}

//------------------------------------------------------------------------------
template<typename T> template<typename G>
void pointFieldHandle<T>::associate(G &g)
{
  g.getFieldsPoint().addArray(this->Name,this->Container);
}

//------------------------------------------------------------------------------
template<typename T> template<typename U>
inline void pointFieldHandle<T>::setExecutionArray(U array)
{
  this->Container.setArrayExecution(array);
}

//------------------------------------------------------------------------------
template<typename T> template<typename U>
inline void pointFieldHandle<T>::setControlArray(U array)
{
  this->Container.setArrayControl(array);
}


//------------------------------------------------------------------------------
template<typename T>
cellFieldHandle<T>::cellFieldHandle(const std::string &name):
  Name(name)
{
}

//------------------------------------------------------------------------------
template<typename T> template<typename G>
void cellFieldHandle<T>::associate(G &g)
{
  g.getFieldsCell().addArray(this->Name,this->Container);
}

//------------------------------------------------------------------------------
template<typename T> template<typename U>
inline void cellFieldHandle<T>::setExecutionArray(U array)
{
  this->Container.setArrayExecution(array);
}

//------------------------------------------------------------------------------
template<typename T> template<typename U>
inline void cellFieldHandle<T>::setControlArray(U array)
{
  this->Container.setArrayControl(array);
}

} }


#endif // __dax_cont_FieldHandles_h
