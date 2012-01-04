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

#include <boost/throw_exception.hpp>
#include <map>

namespace dax { namespace cont { namespace internal {

struct exception_base: virtual std::exception, virtual boost::exception { };
struct array_not_found: virtual exception_base { };

template<typename T>
class FieldContainer : public dax::cont::internal::Object
{
private:
  typedef dax::cont::ArrayPtr<T> ControlArray;
  typedef dax::cont::internal::ArrayContainer<T> ArrayContainer;
  typedef std::map<std::string,ArrayContainer > Map;
  typedef std::pair<std::string,ArrayContainer> MapPair;
  typedef typename Map::iterator MapIterator;
  typedef typename Map::const_iterator ConstMapIterator;


  Map Container;
  //we return this when we can't find a valid array
  ArrayContainer InvalidArray;

public:
  ArrayContainer& get(const std::string &name);
  const ArrayContainer& get(const std::string &name) const;

  bool add(const std::string &name, const ArrayContainer &t);
  bool add(const std::string &name, ControlArray ca);
  bool exists(const std::string &name) const;
  bool remove(const std::string &name);
};

//------------------------------------------------------------------------------
template<typename T>
typename FieldContainer<T>::ArrayContainer& FieldContainer<T>::get(
    const std::string &name)
{
  MapIterator it = this->Container.find(name);
  if(it==this->Container.end())
    {
    return this->InvalidArray;
    }
  return it->second;
}

//------------------------------------------------------------------------------
template<typename T>
const typename FieldContainer<T>::ArrayContainer& FieldContainer<T>::get(
    const std::string &name) const
{
  ConstMapIterator it = this->Container.find(name);
  if(it==this->Container.end())
    {
    return this->InvalidArray;
    }
  return it->second;
}

//------------------------------------------------------------------------------
template<typename T>
bool FieldContainer<T>::add(const std::string &name, const ArrayContainer &t)
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
bool FieldContainer<T>::add(const std::string &name, ControlArray ca)
{
  ArrayContainer container;
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
