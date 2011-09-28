/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_internal_Macros_h
#define __dax_cont_internal_Macros_h

/// Macro to be used to define the class hierarchy.
#define daxTypeMacro(classname, superclassname) \
  public: \
  typedef superclassname Superclass;\
  virtual const char* GetClassName() const { return #classname; }

/// Macro used to to disable copy constructor and assignment operator.
#define daxDisableCopyMacro(classname) \
  private:\
  classname(const classname&); \
  void operator=(const classname&);

/// Macro used to define a class named classnamePtr and classnameWeakPtr of
//types shared_ptr and weak_ptr respectively.
#define daxDefinePtrMacro(classname)\
  typedef boost::shared_ptr<classname> classname##Ptr; \
  typedef boost::weak_ptr<classname> classname##WeakPtr;

#define daxDeclareClass(classname)\
  class classname;\
  daxDefinePtrMacro(classname)


#define daxDefinePtrTemplate1Macro(classname) \
  template<class T> \
  class classname##Ptr : public boost::shared_ptr<classname<T> > \
  { \
  public: \
    classname##Ptr() { } \
    template<class Y> explicit classname##Ptr(Y *p) \
      : boost::shared_ptr<T>(p) { } \
    template<class Y> classname##Ptr(boost::shared_ptr<Y> const &r) \
      : boost::shared_ptr<T>(r) { } \
    template<class Y> explicit classname##Ptr(boost::weak_ptr<Y> const & r) \
      : boost::shared_ptr<T>(r) { } \
  }; \
  template<class T> \
  class classname##WeakPtr : public boost::weak_ptr<classname<T> > \
  { \
  public: \
    classname##WeakPtr() { } \
    template<class Y> explicit classname##WeakPtr(Y *p) \
      : boost::weak_ptr<T>(p) { } \
    template<class Y> classname##WeakPtr(boost::shared_ptr<Y> const &r) \
      : boost::weak_ptr<T>(r) { } \
    template<class Y> explicit classname##WeakPtr(boost::weak_ptr<Y> const & r)\
      : boost::weak_ptr<T>(r) { } \
  };

#define daxDeclareClassTemplate1(classname) \
  template <class T> class classname; \
  daxDefinePtrTemplate1Macro(classname);


/// Error macro to use to report error messages.
#define daxErrorMacro(txt)\
  cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n" txt << endl;

#endif
