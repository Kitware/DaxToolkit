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
#ifndef __dax_exec_internal_ExecutionAdapter_h
#define __dax_exec_internal_ExecutionAdapter_h

#include <dax/Types.h>

#ifdef DAX_DOXYGEN_ONLY

/// \brief Adapter for execution environment.
///
/// This class does not actually exist. Various implementations are created by
/// DeviceAdapters.
///
/// Classes in the execution environment use this class in their template
/// arguments. This class allows the execution environment to adapt to
/// different devices.
///
class ExecutionAdapter
{
public:
  /// This structure contains iterators that can be used to access the arrays
  /// representing fields.  The funny templating of the structure containing
  /// iterators is to handle the case of iterators that are pointers, which
  /// cannot be partially templated (at least before C++11, which is not yet
  /// widely adopted).
  ///
  template <typename T>
  struct FieldStructures
  {
    typedef T *IteratorType;
    typedef const T *IteratorConstType;
  };

  /// This method is used in work objects so that they can raise errors. The
  /// work object will simply call the RaiseError method. This method should
  /// either set the state of the object to signal the error or throw an
  /// exception.
  ///
  void RaiseError(const char *message) const;
};

#endif // DAX_DOXYGEN_ONLY

#endif //__dax_exec_internal_ExecutionAdapter_h
