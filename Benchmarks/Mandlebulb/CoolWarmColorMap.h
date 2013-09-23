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
//  Copyright 2013 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_Benchmarks_Mandlebulb_CoolWarmColorMap_h
#define __dax_Benchmarks_Mandlebulb_CoolWarmColorMap_h

#include <dax/Types.h>

#include <algorithm>

namespace mandle {

class CoolWarmColorMap {
public:
  typedef dax::Tuple<unsigned char, 3> ColorType;

  DAX_CONT_EXPORT CoolWarmColorMap()
  {
    const ColorType *coolWarmTable = GetCoolWarmTable();
    std::copy(coolWarmTable, coolWarmTable+NUM_COLORS, this->Colors);
  }

  DAX_EXEC_CONT_EXPORT
  const ColorType &GetColor(dax::Scalar scalar) const
  {
    if (scalar < 0.0) { scalar = 0.0; }
    if (scalar > 0.9999) { scalar = 0.9999; }
    int index = (int)(NUM_COLORS*scalar);
    return this->Colors[index];
  }

private:
  static const int NUM_COLORS = 33;
  ColorType Colors[NUM_COLORS];

  DAX_CONT_EXPORT
  static const ColorType *GetCoolWarmTable() {
    static const ColorType coolWarmTable[NUM_COLORS] = {
      ColorType(59,76,192),
      ColorType(68,90,204),
      ColorType(77,104,215),
      ColorType(87,117,225),
      ColorType(98,130,234),
      ColorType(108,142,241),
      ColorType(119,154,247),
      ColorType(130,165,251),
      ColorType(141,176,254),
      ColorType(152,185,255),
      ColorType(163,194,255),
      ColorType(174,201,253),
      ColorType(184,208,249),
      ColorType(194,213,244),
      ColorType(204,217,238),
      ColorType(213,219,230),
      ColorType(221,221,221),
      ColorType(229,216,209),
      ColorType(236,211,197),
      ColorType(241,204,185),
      ColorType(245,196,173),
      ColorType(247,187,160),
      ColorType(247,177,148),
      ColorType(247,166,135),
      ColorType(244,154,123),
      ColorType(241,141,111),
      ColorType(236,127,99),
      ColorType(229,112,88),
      ColorType(222,96,77),
      ColorType(213,80,66),
      ColorType(203,62,56),
      ColorType(192,40,47),
      ColorType(180,4,38)
    };
    return coolWarmTable;
  }
};

} // namespace mandle

#endif //__dax_Benchmarks_Mandlebulb_CoolWarmColorMap_h
