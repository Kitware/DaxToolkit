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
#ifndef __dax__cont__GridTags_h
#define __dax__cont__GridTags_h

namespace dax {
namespace cont {
namespace internal
{

/// A tag you can use to identify when grid is an unstructured grid.
///
struct UnstructuredGridTag {  };

/// A subtag of UnstructuredGridTag that specifies the type of cell in the grid
/// through templating.
///
template<class _CellTag>
struct UnstructuredGridOfCell : UnstructuredGridTag { };


/// A tag you can use to identify when a grid is a uniform grid.
///
struct UniformGridTag {  };


/// A tag you can use to state you don't have a grid.
/// Mainly used by algorithms and schedulers to state they work on all grid
/// types
struct UnspecifiedGridTag { };
}
}
}

#endif
