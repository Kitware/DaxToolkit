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
#ifndef __dax_Version_h
#define __dax_Version_h

//
//  DAX_VERSION % 100 is the patch version
//  DAX_VERSION / 100 % 1000 is the minor version
//  DAX_VERSION / 100000 is the major version
//

#define DAX_VERSION 100000

/// \brief DAX_MAJOR_VERSION
///  The macro \p DAX_MAJOR_VERSION encodes the major version number of the
///  Dax toolkit.
#define DAX_MAJOR_VERSION     (DAX_VERSION / 100000)

/// \brief DAX_MINOR_VERSION
///  The macro \p DAX_MINOR_VERSION encodes the minor version number of the
///  Dax toolkit.
#define DAX_MINOR_VERSION     (DAX_VERSION / 100 % 1000)

/// \brief DAX_PATCH_VERSION
///  The macro \p DAX_PATCH_VERSION encodes the patch version number of the
///  Dax toolkit.
#define DAX_PATCH_VERSION     (DAX_VERSION % 100)

#endif