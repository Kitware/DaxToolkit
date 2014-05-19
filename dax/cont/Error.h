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
#ifndef __dax_cont_Error_h
#define __dax_cont_Error_h

// Note that this class and (most likely) all of its subclasses are not
// templated.  If there is any reason to create a Dax control library,
// this class and its subclasses should probably go there.

#include <string>

namespace dax {
namespace cont {

/// The superclass of all exceptions thrown by any Dax function or method.
///
class Error
{
public:
  const std::string &GetMessage() const { return this->Message; }

#if defined(_WIN32) && defined(_MSC_VER)
  const std::string &GetMessageA() const { return this->Message; }
  const std::string &GetMessageW() const { return this->Message; }
#endif

protected:
  Error() { }
  Error(const std::string message) : Message(message) { }

  void SetMessage(const std::string &message) {
    this->Message = message;
  }

private:
  std::string Message;
};

}
} // namespace dax::cont

#endif //__dax_cont_Error_h
