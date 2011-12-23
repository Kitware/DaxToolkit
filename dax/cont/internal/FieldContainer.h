/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_internal_FieldContainer_h
#define __dax_cont_internal_FieldContainer_h

#include <dax/cont/internal/Object.h>
#include <dax/cont/Array.h>
#include <dax/cont/internal/ArrayContainer.h>
#include <map>


namespace dax { namespace cont { namespace internal {
template<typename T>
class FieldContainer : public dax::cont::internal::Object
{
private:
  typedef dax::cont::ArrayPtr<T> ContArray;
  typedef dax::cont::internal::ArrayContainer<T> TArray;
  typedef std::pair<std::string,TArray> MapPair;
  std::map<std::string,TArray> Container;

public:
  TArray& get(const std::string &name)
    {return Container.find(name)->second;}
  const TArray& get(const std::string &name) const
    {return Container.find(name)->second;}

  bool add(const std::string &name, const TArray &t);
  bool add(const std::string &name, ContArray ca);
  bool exists(const std::string &name) const;
  bool remove(const std::string &name);
};

//------------------------------------------------------------------------------
template<typename T>
bool FieldContainer<T>::add(const std::string &name, const TArray &t)
{
  if(this->exists(name))
    {
    return false;
    }
  Container.insert(MapPair(name,t));
  return true;
}

//------------------------------------------------------------------------------
template<typename T>
bool FieldContainer<T>::add(const std::string &name, ContArray ca)
{
  TArray container;
  container.setArrayControl(ca);
  return this->add(name,container);
}

//------------------------------------------------------------------------------
template<typename T>
bool FieldContainer<T>::remove(const std::string &name)
{
  if(this->exists(name))
    {
    Container.erase(name);
    return true;
    }
  return false;
}

//------------------------------------------------------------------------------
template<typename T>
inline bool FieldContainer<T>::exists(const std::string &name) const
{
  return Container.find(name) != Container.end();
}

} } }

#endif // __dax_cont_internal_FieldContainer_h
