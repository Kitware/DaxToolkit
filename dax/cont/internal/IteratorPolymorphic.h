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
class IteratorPolymorphicLValueDelegate
{
public:
  virtual void Set(const T &value) = 0;
  virtual T Get() const = 0;
  virtual ~IteratorPolymorphicLValueDelegate() { }
};

//-----------------------------------------------------------------------------
template<class IteratorType>
class IteratorPolymorphicLValueDelegateImplementation
    : public IteratorPolymorphicLValueDelegate<
        typename std::iterator_traits<IteratorType>::value_type>
{
public:
  typedef typename std::iterator_traits<IteratorType>::value_type ValueType;

  IteratorPolymorphicLValueDelegateImplementation(IteratorType iter)
    : Target(iter) { }
  virtual void Set(const ValueType &value){
    *this->Target = value;
  }
  virtual ValueType Get() const {
    return *this->Target;
  }

private:
  IteratorType Target;
  IteratorPolymorphicLValueDelegateImplementation(); // Not implemented
};

//-----------------------------------------------------------------------------
template<typename T>
class IteratorPolymorphicLValue
{
public:
  template<class IteratorType>
  IteratorPolymorphicLValue(IteratorType iterator)
    : Delegate(
        new IteratorPolymorphicLValueDelegateImplementation<IteratorType>(
          iterator)) { }

  IteratorPolymorphicLValue &operator=(const T &rvalue) {
    this->Delegate->Set(rvalue);
    return *this;
  }

  operator T() const {
    return this->Delegate->Get();
  }

private:
  boost::shared_ptr<IteratorPolymorphicLValueDelegate<T> > Delegate;
};

//-----------------------------------------------------------------------------
template<typename T>
class IteratorPolymorphicDelegate
{
public:
  virtual void Increment() = 0;
  virtual void Decrement() = 0;
  virtual void Advance(dax::Id n) = 0;
  virtual bool Equal(const IteratorPolymorphicDelegate<T> *other) const = 0;
  virtual IteratorPolymorphicLValue<T> Dereference() const = 0;

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
  virtual bool Equal(const IteratorPolymorphicDelegate<ValueType> *other) const
  {
    typedef IteratorPolymorphicDelegateImplementation<IteratorType> MyType;
    const MyType *otherCast = dynamic_cast<const MyType *>(other);
    return (otherCast && (this->Target == otherCast->Target));
  }
  virtual IteratorPolymorphicLValue<ValueType> Dereference() const {
    IteratorPolymorphicLValue<ValueType> lvalue(this->Target);
    return lvalue;
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
        IteratorPolymorphicLValue<T>,
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
  bool equal(const IteratorPolymorphic<T> &other) const {
    return this->Delegate->Equal(other.Delegate.get());
  }

  IteratorPolymorphicLValue<T> dereference() const {
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
