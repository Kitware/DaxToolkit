/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxMacros_h
#define __daxMacros_h

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


/// Error macro to use to report error messages.
#define daxErrorMacro(txt)\
  cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n" txt << endl;

#endif
