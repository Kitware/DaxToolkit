/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_internal_IteratorPolymorphic_h
#define __dax_cont_internal_IteratorPolymorphic_h

#include <dax/Types.h>

#include <boost/iterator/iterator_facade.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>

namespace dax {
namespace cont {
namespace internal {

//-----------------------------------------------------------------------------
template<typename T>
class IteratorPolymorphicDelegate
{
public:
  virtual void Increment() = 0;
  virtual void Decrement() = 0;
  virtual void Advance(dax::Id n) = 0;
  virtual dax::Id DistanceTo(
      const IteratorPolymorphicDelegate<T> *other) const = 0;
  virtual bool Equal(const IteratorPolymorphicDelegate<T> *other) const = 0;
  virtual T & Dereference() const = 0;

  virtual IteratorPolymorphicDelegate<T> *MakeCopy() const = 0;

  virtual ~IteratorPolymorphicDelegate() { }

protected:
  IteratorPolymorphicDelegate() { }

private:
  IteratorPolymorphicDelegate(const IteratorPolymorphicDelegate<T> &); // Not implemented
  void operator=(const IteratorPolymorphicDelegate<T> &); // Not implemented
};

//-----------------------------------------------------------------------------
template<class IteratorType>
class IteratorPolymorphicDelegateImplementation
    : public IteratorPolymorphicDelegate<
        typename std::iterator_traits<IteratorType>::value_type>
{
public:
  typedef typename std::iterator_traits<IteratorType>::value_type ValueType;

  IteratorPolymorphicDelegateImplementation(IteratorType iter)
    : Target(iter) { }

  virtual void Increment() { this->Target++; }
  virtual void Decrement() { this->Target--; }
  virtual void Advance(dax::Id n) { this->Target += n; }
  virtual dax::Id DistanceTo(
      const IteratorPolymorphicDelegate<ValueType> *other) const {
    typedef IteratorPolymorphicDelegateImplementation<IteratorType> MyType;
    const MyType *otherCast = dynamic_cast<const MyType *>(other);
    if (!otherCast)
      {
      // Error condition.
      return -1;
      }
    return otherCast->Target - this->Target;
  }
  virtual bool Equal(const IteratorPolymorphicDelegate<ValueType> *other) const
  {
    typedef IteratorPolymorphicDelegateImplementation<IteratorType> MyType;
    const MyType *otherCast = dynamic_cast<const MyType *>(other);
    return (otherCast && (this->Target == otherCast->Target));
  }
  virtual ValueType &Dereference() const {
    return *this->Target;
  }

  virtual IteratorPolymorphicDelegate<ValueType> *MakeCopy() const {
    IteratorPolymorphicDelegateImplementation<IteratorType> *copy
        = new IteratorPolymorphicDelegateImplementation<IteratorType>(
            this->Target);
    return copy;
  }
private:
  IteratorType Target;

  IteratorPolymorphicDelegateImplementation();  // Not implemented.
};

//-----------------------------------------------------------------------------
template<typename T>
class IteratorPolymorphic
    : public boost::iterator_facade<
        IteratorPolymorphic<T>,
        T,
        boost::random_access_traversal_tag,
        T &,
        dax::Id>
{
public:
  IteratorPolymorphic() : Delegate(NULL) { }
  template<class IteratorType>
  IteratorPolymorphic(IteratorType iterator)
    : Delegate(
        new IteratorPolymorphicDelegateImplementation<IteratorType>(
          iterator)) { }
  IteratorPolymorphic(const IteratorPolymorphic<T> &src)
    : Delegate(src.Delegate->MakeCopy()) { }
  void operator=(const IteratorPolymorphic<T> &src) {
    this->Delegate.reset(src.Delegate->MakeCopy());
  }

private:
  boost::scoped_ptr<IteratorPolymorphicDelegate<T> > Delegate;

  // Implementations for boost iterator_facade.
  friend class boost::iterator_core_access;
  void increment() { this->Delegate->Increment(); }
  void decrement() { this->Delegate->Decrement(); }
  void advance(dax::Id n) { this->Delegate->Advance(n); }
  dax::Id distance_to(const IteratorPolymorphic<T> &other) const {
    return this->Delegate->DistanceTo(other.Delegate.get());
  }

  bool equal(const IteratorPolymorphic<T> &other) const {
    return this->Delegate->Equal(other.Delegate.get());
  }

  T &dereference() const {
    return this->Delegate->Dereference();
  }
};

template<class IteratorType>
IteratorPolymorphic<typename std::iterator_traits<IteratorType>::value_type>
make_IteratorPolymorphic(IteratorType iterator)
{
  typedef typename std::iterator_traits<IteratorType>::value_type ValueType;
  IteratorPolymorphic<ValueType> iteratorPolymorphic(iterator);
  return iteratorPolymorphic;
}

}
}
}

#endif // __dax_cont_internal_IteratorPolymorphic_h
