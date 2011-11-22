#ifndef COMPILETIMEASSERT_H
#define COMPILETIMEASSERT_H

/*
  From Modern C++ Design
*/

namespace
{
  //true case has a constructor that takes
  //anything so that it passes when we pass it
  //the struct whose name is the assert error message
  template<bool> struct CompileTimeAssert
  {
    CompileTimeAssert(...);
  };

  //false case doesn't have a constructor so
  //we get an error message with our assert message
  //Plus this class will never be compiled into the source
  //code so it is totally free
  template<> struct CompileTimeAssert<false>;

  //Older style C style compile time assert
  //by using a zero length array
  #define CStyleCompileTimeAssert(expr,message){char ERROR_##message[ (expr)==true ? 1 : -1]; }
}

//Macro that makes calling the CompileTimeAssert alot easier
//first argument is the expression to confirm is true, and the
//second is the error message
//CStyleCompileTimeAssert is used as a fail safe
#define DAX_ASSERT(expression,error_message) \
{ \
  class ERROR_##error_message{};\
  CompileTimeAssert<(expression)==true>( (ERROR_##error_message()) );\
  CStyleCompileTimeAssert(expression,error_message) \
}

#endif
