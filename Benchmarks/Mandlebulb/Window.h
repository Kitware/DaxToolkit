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

#include "Mandlebulb.h"

#include <iostream>


namespace
{
GLvoid* bufferObjectPtr( unsigned int idx )
  {
  return (GLvoid*) ( ((char*)NULL) + idx );
  }
}
namespace mandle{

/// \brief Render Window for mandlebulb
///
/// Bare-bones class that fulfullis the requirements of WindowBase but
/// has no ability to interact with opengl other than to close down the window
///
///
class Window : public dax::opengl::testing::WindowBase<Window>
{
public:
  DAX_CONT_EXPORT Window()
  {
    CurrentTime = 0;
    PreviousTime = 0;
    Fps = 0;
    PreviousFps = 0;
    FrameCount = 0;

    MouseX = 0;
    MouseY = 0;
    ActiveMouseButtons = 0;

    RotateX = 0;
    RotateY = 0;
    TranslateZ = -3;

    //set this first buffer to be the active buffer
    BufferCount = 0;
    BufferHandles =  FirstBuffer;
    Iteration = 1;
    Remesh = true;
    Mode = 0;

    //compute the mandlebulb
    this->Info = computeMandlebulb( );
  }

  //called after opengl is inited
  DAX_CONT_EXPORT void PostInit()
  {
  }

  DAX_CONT_EXPORT void Display()
  {
  if(this->Remesh)
    {

    if ( this->BufferCount%2==0 )
      { this->BufferHandles = this->FirstBuffer; }
    else
      { this->BufferHandles = this->SecondBuffer; }

    if(this->Mode==0)
      this->MandleSurface = extractSurface(this->Info,this->Iteration);
    else
      {
      this->MandleSurface = extractSlice(this->Info,this->Iteration);
      }
    bindSurface(this->MandleSurface, this->BufferHandles );

    this->Remesh = false;
    }

  // Clear Color and Depth Buffers
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  //Move the camera
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.0, 0.0, this->TranslateZ);
  glRotatef(this->RotateX, 1.0, 0.0, 0.0);
  glRotatef(this->RotateY, 0.0, 1.0, 0.0);

  glEnableClientState(GL_VERTEX_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, this->BufferHandles[0]);
  glVertexPointer(3, GL_FLOAT, 0, bufferObjectPtr(0) );

  glEnableClientState(GL_COLOR_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, this->BufferHandles[1]);
  glColorPointer(4, GL_UNSIGNED_BYTE, 0, bufferObjectPtr(0) );

  glEnableClientState(GL_NORMAL_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, this->BufferHandles[2]);
  glNormalPointer(GL_FLOAT, 0, bufferObjectPtr(0) );

  glDrawArrays( GL_TRIANGLES, 0, this->MandleSurface.Data.GetNumberOfPoints() );

  glDisableClientState(GL_NORMAL_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  //display the fps
  if(this->Fps > this->PreviousFps+1 || this->Fps < this->PreviousFps-1 )
    {
    std::cout << "fps: " << this->Fps << std::endl;
    std::cout << "triangles: " << this->MandleSurface.Data.GetNumberOfCells() << std::endl;
    this->Fps = this->PreviousFps;
    }

  //terminate once we have been running for ~10 secs
  if(this->CurrentTime >= 10000)
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
   if(key == 27) //escape pressed
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

private:
  MandlebulbInfo Info;
  MandlebulbSurface MandleSurface;
  dax::Scalar Iteration;
  bool Remesh;
  int Mode; //0 marching cubes, 1 slice

  GLuint FirstBuffer[4];
  GLuint SecondBuffer[4];
  GLuint* BufferHandles;
  int BufferCount;

  float CurrentTime;
  float PreviousTime;
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
