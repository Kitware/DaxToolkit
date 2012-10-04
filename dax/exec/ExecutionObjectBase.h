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

#ifndef __dax_exec_ExecutionObjectBase_h
#define __dax_exec_ExecutionObjectBase_h

///Base ExecutionObjectBase for execution objects to inherit from so that you can
///use an arbitrary object as a parameter to a worklet. Any method you want
///to use on the execution side must be have the DAX_EXEC_EXPORT
namespace dax { namespace exec {
class ExecutionObjectBase
{
};
} }

#endif // __dax_exec_ExecutionObjectBase_h
