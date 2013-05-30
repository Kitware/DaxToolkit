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
#ifndef __dax__opengl__testing__TestingWindow_h
#define __dax__opengl__testing__TestingWindow_h

#include <dax/internal/ExportMacros.h>
#include <dax/opengl/testing/WindowBase.h>
namespace dax{
namespace opengl{
namespace testing{

/// \brief Basic Render Window that only makes sure opengl has a valid context
///
/// Bare-bones class that fulfullis the requirements of WindowBase but
/// has no ability to interact with opengl other than to close down the window
///
///
class TestingWindow : public dax::opengl::testing::WindowBase<TestingWindow>
{
public:
  DAX_CONT_EXPORT TestingWindow(){};

  DAX_CONT_EXPORT void Display()
  {}

  DAX_CONT_EXPORT void Idle()
  {}

  DAX_CONT_EXPORT void ChangeSize(int daxNotUsed(w), int daxNotUsed(h) )
  {}

  DAX_CONT_EXPORT void Key(unsigned char key,
                           int daxNotUsed(x), int daxNotUsed(y) )
  {
   if(key == 27) //escape pressed
    {
    exit(0);
    }
  }

  DAX_CONT_EXPORT void SpecialKey(int daxNotUsed(key),
                                  int daxNotUsed(x), int daxNotUsed(y) )
  {}

  DAX_CONT_EXPORT void Mouse(int daxNotUsed(button), int daxNotUsed(state),
                             int daxNotUsed(x), int daxNotUsed(y) )
  {}

  DAX_CONT_EXPORT void MouseMove(int daxNotUsed(x), int daxNotUsed(y) )
  {}
};


}
}
}
#endif
