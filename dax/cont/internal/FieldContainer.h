
#ifndef __dax_cont_internal_FieldContainer_h
#define __dax_cont_internal_FieldContainer_h

#include <dax/cont/internal/Object.h>
#include <dax/cont/Array.h>
#include <map>


namespace dax { namespace cont { namespace internal {
template<typename T>
class FieldContainer : public dax::cont::internal::Object
{
private:
  typedef dax::cont::Array<T> TArray;
  typedef std::pair<std::string,TArray*> MapPair;

  std::map<std::string,TArray*> Container;

public:
  TArray& get(const std::string &name)
    {
    return *(Container.find(name)->second);
    }

  const TArray& get(const std::string &name) const
    {
    return *(Container.find(name)->second);
    }

  bool add(const std::string &name, TArray *t)
    {
    if(this->exists(name))
      {
      return false;
      }
    Container.insert(MapPair(name,t));
    return true;
    }

  bool remove(const std::string &name)
    {
    if(this->exists(name))
      {
      Container.erase(name);
      return true;
      }
    return false;
    }

  bool exists(const std::string &name)
    {
    return Container.find(name) != Container.end();
    }
};

} } }

#endif // __dax_cont_internal_FieldContainer_h
