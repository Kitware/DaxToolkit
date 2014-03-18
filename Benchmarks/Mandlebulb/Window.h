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
#ifndef __dax__benchmarks_Mandlebulb_ManblebulbWindow_h
#define __dax__benchmarks_Mandlebulb_ManblebulbWindow_h

#include <dax/internal/ExportMacros.h>
#include <dax/opengl/testing/WindowBase.h>

#include "ArgumentsParser.h"
#include "Mandlebulb.h"
#include "ShaderProgram.h"

namespace mandle {

//-----------------------------------------------------------------------------
struct TwizzledGLHandles
{
  //stores two sets of handles, each time
  //you call for the handles it switches
  //what ones it returns
  TwizzledGLHandles()
    {
    this->State = 0;
    }

  ~TwizzledGLHandles()
    {
    this->release();
    }

  void release()
  {
    glDeleteBuffers(2,CoordHandles);
    glDeleteBuffers(2,ColorHandles);
    glDeleteBuffers(2,NormHandles);
  }

  void initHandles()
  {
    glGenBuffers(2,CoordHandles);
    glGenBuffers(2,ColorHandles);
    glGenBuffers(2,NormHandles);

    for(int i=0; i < 2; ++i)
    {
      glBindBuffer(GL_ARRAY_BUFFER, CoordHandles[i]);
      glBindBuffer(GL_ARRAY_BUFFER, ColorHandles[i]);
      glBindBuffer(GL_ARRAY_BUFFER, NormHandles[i]);
    }
  }

  void handles(GLuint& coord, GLuint& color, GLuint& norm) const
    {
    coord = CoordHandles[this->State%2];
    color = ColorHandles[this->State%2];
    norm  = NormHandles[this->State%2];
    }

  void switchHandles()
    {
    ++this->State;
    }

private:
  GLuint CoordHandles[2];
  GLuint ColorHandles[2];
  GLuint NormHandles[2];
  unsigned int State; //when we roll over we need to stay positive
};


/// \brief Render Window for display the mandlebulb demo
///
/// Interactive render window for the mandlebulb demo.
///
//-----------------------------------------------------------------------------
class Window : public dax::opengl::testing::WindowBase<Window>
{
public:
  Window(const ArgumentsParser &arguments);
  virtual ~Window();

  //properly release all gpu and cuda memory
  void Cleanup();

  //controls if we should auto exit
  void CheckMaxTime();

  //increment to the demo based on the current time
  void NextAutoPlayStep();

  //display updated fps
  void DisplayCurrentFPS();

  //remesh the mandlebulb given current iso value and mode
  void RemeshSurface();

  //methods to fulfill the WindowBase Reqs
  void PostInit();
  void Display();
  void Idle();
  void ChangeSize(int,int);
  void Key(unsigned char, int, int);
  void SpecialKey(int,int,int);
  void Mouse(int,int,int,int);
  void MouseMove(int,int);
  void PassiveMouseMove(int,int);

private:
  //move ManblebulbInfo and MandlebublSurface to be implementation
  //details
  class InternalMandleData;
  InternalMandleData* MandleData;

  dax::Scalar Iteration; //tracks the escape iteration to contour on
  dax::Scalar ClipRatio; //tracks where to clip at
  bool Remesh;
  int Mode; //0 marching cubes, 1 clip


  //gl array ids that hold the rendering info
  TwizzledGLHandles TwizzleHandles;

  //shader program that holds onto all the shader details
  mandle::ShaderProgram Shaders;

  //holds the location to upload the current time too, for the shader
  GLint ShaderLightLocation;

  //time tracking vars
  float CurrentTime;
  float PreviousTime;
  float MaxTime; //used to see when we should exit

  //fps tracking vars
  float Fps;
  float PreviousFps;
  int FrameCount;

  //mouse tracking
  int MouseX;
  int MouseY;
  int ActiveMouseButtons;

  //camera rotation tracking
  float RotateX;
  float RotateY;
  float TranslateZ;

  //auto play demo tracking
  bool AutoPlay;

  //size of each dimension of the grid
  dax::Id GridSize;
};

}

#endif
