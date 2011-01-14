/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __fncMacros_h
#define __fncMacros_h

// Macro to be used to define the class hierarchy.
#define fncTypeMacro(classname, superclassname) \
  public: \
  typedef superclassname Superclass;\
  virtual const char* GetClassName() const { return #classname; }


#define fncDisableCopyMacro(classname) \
  private:\
  classname(const classname&); \
  void operator=(const classname&);


#define fncDefinePtrMacro(classname)\
  typedef boost::shared_ptr<classname> classname##Ptr; \
  typedef boost::weak_ptr<classname> classname##WeakPtr;

#endif
