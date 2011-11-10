#ifndef  HETEROGENEOUSCONTAINER_H
#define HETEROGENEOUSCONTAINER_H

#include <typeinfo>
#include <list>
#include <exception>

/* From
C++ Design. Strategies for intuitive and error resistant software.
Tim Bailey. Tech. Report, 2005-2006.
*/
typedef std::string HContainerError;

class ObjectHandle {
public:
  virtual ~ObjectHandle() {}
  virtual const std::type_info& type() const = 0;
};

template<typename T>
class ObjectHandleT : public ObjectHandle {
public:
  ObjectHandleT(const T &t) : obj(t) {}
  const std::type_info& type() const { return typeid(T); }
  const T& get() { return obj; }
private:
  T obj;
};

class HQueue {
public:
  HQueue() {}
  ~HQueue()
  {
    ObjQueue::iterator i;
    for (i = data.begin(); i!= data.end(); ++i)
      delete *i;
  }
  template<typename T>
  void push(const T &t)
  {
    data.push_back(new ObjectHandleT<T>(t));
  }
  template<typename T>
  void pop(T &t)
  {
    if (!is_type<T>())
      throw HContainerError("HQueue Type Mismatch");
    t = static_cast<ObjectHandleT<T>*>(data.front())->get();
    delete data.front();
    data.pop_front();
  }
  template<typename T>
  bool is_type() const { return typeid(T) == data.front()->type(); }
  template<typename T>
  bool is_type(const T &t) const { return is_type<T>(); }
  std::size_t size() const { return data.size(); }
private:
  HQueue(const HQueue& rhs);
  HQueue& operator=(const HQueue& rhs);
  // prevent copying
  // prevent assignment
  typedef std::list<ObjectHandle*> ObjQueue;
  ObjQueue data;
};

#endif
