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
#ifndef __dax_CellTag_h
#define __dax_CellTag_h

namespace dax {

#ifdef DAX_DOXYGEN_ONLY
/// A CellTag object is an empty struct used to identify a particular cell
/// type. It is used as a template parameter for classes and functions that can
/// be specialized on cell type. All cell tags follow the naming convention of
/// CellTag* where the * appendix is the name of the cell type.
///
struct CellTag___ {  };
#endif // DAX_DOXYGEN_ONLY

struct CellTagHexahedron { };
struct CellTagLine { };
struct CellTagQuadrilateral { };
struct CellTagTetrahedron { };
struct CellTagTriangle { };
struct CellTagVertex { };
struct CellTagVoxel { };
struct CellTagWedge { };

} // namespace dax

#endif //__dax_CellTag_h
