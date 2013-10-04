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
#include <dax/Extent.h>
#include <dax/math/Compare.h>

#include "ArgumentsParser.h"
#include "Mandlebulb.h"
#include "ShaderCode.h"
#include "ShaderProgram.h"

#include <iostream>


namespace
{
GLvoid* bufferObjectPtr( unsigned int idx )
  {
  return (GLvoid*) ( ((char*)NULL) + idx );
  }
}

namespace mandle{


struct TwizzledGLHandles
{
  //stores two sets of handles, each time
  //you call for the handles it switches
  //what ones it returns
  TwizzledGLHandles()
    {
    this->State = 0;
    }

  void initHandles()
  {
    glGenBuffers(2,CoordHandles);
    glGenBuffers(2,ColorHandles);
    glGenBuffers(2,NormHandles);


    glBindBuffer(GL_ARRAY_BUFFER, CoordHandles[0]);
    glBindBuffer(GL_ARRAY_BUFFER, CoordHandles[1]);
    glBindBuffer(GL_ARRAY_BUFFER, ColorHandles[0]);
    glBindBuffer(GL_ARRAY_BUFFER, ColorHandles[1]);
    glBindBuffer(GL_ARRAY_BUFFER, NormHandles[0]);
    glBindBuffer(GL_ARRAY_BUFFER, NormHandles[1]);

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


/// \brief Render Window for mandlebulb
///
/// Bare-bones class that fulfullis the requirements of WindowBase but
/// has no ability to interact with opengl other than to close down the window
///
///
class Window : public dax::opengl::testing::WindowBase<Window>
{
public:
  DAX_CONT_EXPORT Window(const ArgumentsParser &arguments)
  {
    this->CurrentTime = 0;
    this->PreviousTime = 0;
    this->MaxTime = (float)arguments.time();
    this->Fps = 0;
    this->PreviousFps = 0;
    this->FrameCount = 0;

    this->MouseX = 0;
    this->MouseY = 0;
    this->ActiveMouseButtons = 0;

    this->RotateX = 0;
    this->RotateY = 0;
    this->TranslateZ = -3;

    this->Iteration = 1;
    this->Remesh = true;
    this->Mode = 0;
  }

  virtual ~Window()
  {
  }

  //called after opengl is inited
  DAX_CONT_EXPORT void PostInit()
  {
    //init the gl handles that the twizzler holds
    this->TwizzleHandles.initHandles();

    //build up our shaders
    this->ShaderProgram.add_vert_shader(make_vertex_shader_code());
    this->ShaderProgram.add_frag_shader(make_fragment_shader_code());
    this->ShaderProgram.build();

    glEnable(GL_DEPTH);
    glEnable(GL_DEPTH_TEST);

    //clear the render window
    glClearColor(1.0, 1.0, 1.0, 1.0);

    //compute the mandlebulb
    this->Info = computeMandlebulb( );

    if(this->Mode==0)
      this->MandleSurface = extractSurface(this->Info,this->Iteration);
    else
      this->MandleSurface = extractSlice(this->Info,this->Iteration);

    GLuint coord, color, norm;

    this->TwizzleHandles.handles(coord,color,norm);
    bindSurface(this->MandleSurface, coord, color, norm );

    this->Remesh = false;
  }

  DAX_CONT_EXPORT void Display()
  {
  GLuint coord, color, norm;
  if(this->Remesh)
    {
    this->TwizzleHandles.switchHandles();
    this->TwizzleHandles.handles(coord,color,norm);

    if(this->Mode==0)
      this->MandleSurface = extractSurface(this->Info,this->Iteration);
    else
      this->MandleSurface = extractSlice(this->Info,this->Iteration);

    bindSurface(this->MandleSurface, coord, color, norm );

    this->Remesh = false;
    }
  else
    {
    this->TwizzleHandles.handles(coord,color,norm);
    }

  // Clear Color and Depth Buffers
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  //bind the shaders
  glUseProgram(this->ShaderProgram.program_id());

  //Move the camera
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.0, 0.0, this->TranslateZ);
  glRotatef(this->RotateX, 1.0, 0.0, 0.0);
  glRotatef(this->RotateY, 0.0, 1.0, 0.0);

  glEnableClientState(GL_VERTEX_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, coord);
  glVertexPointer(3, GL_FLOAT, 0, bufferObjectPtr(0) );

  glEnableClientState(GL_COLOR_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, color);
  glColorPointer(4, GL_UNSIGNED_BYTE, 0, bufferObjectPtr(0) );

  glEnableClientState(GL_NORMAL_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, norm);
  glNormalPointer(GL_FLOAT, 0, bufferObjectPtr(0) );

  glDrawArrays( GL_TRIANGLES, 0, this->MandleSurface.Data.GetNumberOfPoints() );

  glDisableClientState(GL_NORMAL_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  //unbind the shaders
  glUseProgram(0);

  //display the fps
  if(this->Fps > this->PreviousFps+1 || this->Fps < this->PreviousFps-1 )
    {
    std::cout << "fps: " << this->Fps << std::endl;
    std::cout << "triangles: " << this->MandleSurface.Data.GetNumberOfCells() << std::endl;
    this->Fps = this->PreviousFps;
    }

  //terminate once we have been running for ~10 secs
  if((this->MaxTime > 0)
     && (this->CurrentTime >= 10000))
    {
    exit(0);
    }

  glutSwapBuffers();
  }

  DAX_CONT_EXPORT void Idle()
  {
  //calculate FPS
  ++this->FrameCount;
  //  Get the number of milliseconds since glutInit called
  //  (or first call to glutGet(GLUT ELAPSED TIME)).
  CurrentTime = glutGet(GLUT_ELAPSED_TIME);
  //  Calculate time passed
  int timeInterval = this->CurrentTime - this->PreviousTime;
  if(timeInterval > 1000)
    {
    this->Fps = this->FrameCount / (timeInterval / 1000.0f);
    this->PreviousTime = this->CurrentTime;
    this->FrameCount = 0;
    }

  glutPostRedisplay();
  }

  DAX_CONT_EXPORT void ChangeSize(int w, int h)
  {
    h = std::max(h,1);
    float ratio =  w * 1.0 / h;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, w, h);
    gluPerspective(45.0f, ratio, 0.01f, 10.0f);
    glMatrixMode(GL_MODELVIEW);
  }

  DAX_CONT_EXPORT void Key(unsigned char key,
                           int daxNotUsed(x), int daxNotUsed(y) )
  {
   if ((key == 27) //escape pressed
       || (key == 'q'))
    {
    exit(0);
    }
  }

  DAX_CONT_EXPORT void SpecialKey(int key,
                                  int daxNotUsed(x), int daxNotUsed(y) )
  {
  switch (key)
    {
    case GLUT_KEY_UP:
      if(this->Iteration < 30)
        {
        this->Iteration += 1;
        this->Remesh = true;
        }
      break;
    case GLUT_KEY_DOWN :
      if(this->Iteration > 0)
        {
        this->Iteration -= 1;
        this->Remesh = true;
        }
      break;
    case GLUT_KEY_LEFT:
      this->Mode = 0;
      this->Remesh = true;
      break;
    case GLUT_KEY_RIGHT:
      this->Mode = 1;
      this->Remesh = true;
      break;
    default:
      break;
    }
  }

  DAX_CONT_EXPORT void Mouse(int button, int state, int x, int y )
  {
  if (state == GLUT_DOWN)
    {
    this->ActiveMouseButtons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
    this->ActiveMouseButtons = 0;
    }

    this->MouseX = x;
    this->MouseY = y;
  }

  DAX_CONT_EXPORT void MouseMove(int x, int y )
  {
    float dx = (float)(x - MouseX);
    float dy = (float)(y - MouseY);

    if (ActiveMouseButtons & 1)
    {
        this->RotateX += dy * 0.2f;
        this->RotateY += dx * 0.2f;
    }
    else if (ActiveMouseButtons & 4)
    {
        this->TranslateZ += dy * 0.01f;
    }

    this->MouseX = x;
    this->MouseY = y;
  }


  DAX_CONT_EXPORT void PassiveMouseMove(int,int)
  { }

private:
  MandlebulbInfo Info;
  MandlebulbSurface MandleSurface;
  dax::Scalar Iteration;
  bool Remesh;
  int Mode; //0 marching cubes, 1 slice

  //gl array ids that hold the rendering info
  TwizzledGLHandles TwizzleHandles;

  //shader program that holds onto all the shader details
  mandle::ShaderProgram ShaderProgram;

  float CurrentTime;
  float PreviousTime;
  float MaxTime;
  float Fps;
  float PreviousFps;
  int FrameCount;

  int MouseX;
  int MouseY;
  int ActiveMouseButtons;

  float RotateX;
  float RotateY;
  float TranslateZ;
};


}

#endif
