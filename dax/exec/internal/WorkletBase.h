//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_exec_WorkletBase_h
#define __dax_exec_WorkletBase_h

#include <dax/Types.h>

#include <dax/cont/arg/ExecObject.h>
#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/Tag.h>
#include <dax/cont/sig/WorkId.h>
#include <dax/exec/internal/ErrorMessageBuffer.h>

namespace dax {
namespace exec {
namespace internal {

/// Base class for all worklet classes. Worklet classes are subclasses and a
/// operator() const is added to implement an algorithm in Dax. Different
/// worklets have different calling semantics.
///
class WorkletBase
{
public:
  DAX_EXEC_CONT_EXPORT WorkletBase() {  }

  DAX_EXEC_EXPORT void RaiseError(const char *message) const
  {
    this->ErrorMessage.RaiseError(message);
  }

  /// Set the error message buffer so that running algorithms can report
  /// errors. This is supposed to be set by the dispatcher. This method may be
  /// replaced as the execution semantics change.
  ///
  DAX_CONT_EXPORT void SetErrorMessageBuffer(
      const dax::exec::internal::ErrorMessageBuffer &buffer)
  {
    this->ErrorMessage = buffer;
  }

protected:
  typedef dax::cont::sig::placeholders::_1 _1;
  typedef dax::cont::sig::placeholders::_2 _2;
  typedef dax::cont::sig::placeholders::_3 _3;
  typedef dax::cont::sig::placeholders::_4 _4;
  typedef dax::cont::sig::placeholders::_5 _5;
  typedef dax::cont::sig::placeholders::_6 _6;
  typedef dax::cont::sig::placeholders::_7 _7;
  typedef dax::cont::sig::placeholders::_8 _8;
  typedef dax::cont::sig::placeholders::_9 _9;
  typedef dax::cont::sig::In In;
  typedef dax::cont::sig::Out Out;
  typedef dax::cont::sig::WorkId WorkId;

#ifndef UserObject
# define UserObject dax::cont::arg::ExecObject(*)()
#endif

private:
  dax::exec::internal::ErrorMessageBuffer ErrorMessage;
};


}
}
} // namespace dax::exec::internal

#endif //__dax_exec_WorkletBase_h
