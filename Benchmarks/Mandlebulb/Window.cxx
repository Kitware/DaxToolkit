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

#include "Window.h"
#include "ShaderCode.h"

namespace
{
GLvoid* bufferObjectPtr( unsigned int idx )
  {
  return (GLvoid*) ( ((char*)NULL) + idx );
  }
}

namespace mandle {

class Window::InternalMandleData
{
public:
  MandlebulbVolume Volume;
  MandlebulbSurface Surface;
};


//-----------------------------------------------------------------------------
Window::Window(const ArgumentsParser &arguments)
{
  this->MandleData = new mandle::Window::InternalMandleData();

  this->AutoPlay = arguments.AutoPlay();

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

//-----------------------------------------------------------------------------
Window::~Window()
{
  delete this->MandleData;
}

//-----------------------------------------------------------------------------
void Window::CheckMaxTime() const
{
  //terminate once we have been running for ~10 secs
  if((this->MaxTime > 0) && (this->CurrentTime >= 10000))
    {
    exit(0);
    }
}

//-----------------------------------------------------------------------------
void Window::NextAutoPlayStep()
{
  if(this->AutoPlay)
    {
    this->Iteration += 0.1;
    this->Remesh = true;
    }
}

//-----------------------------------------------------------------------------
void Window::DisplayCurrentFPS()
{
  if(this->Fps > this->PreviousFps+1 || this->Fps < this->PreviousFps-1 )
    {
    std::cout << "fps: " << this->Fps << std::endl;
    std::cout << "triangles: " <<
                 this->MandleData->Surface.Data.GetNumberOfCells() << std::endl;
    this->Fps = this->PreviousFps;
    }
}

//-----------------------------------------------------------------------------
void Window::RemeshSurface()
{
  if(this->Remesh)
    {
    GLuint coord, color, norm;
    this->TwizzleHandles.switchHandles();
    this->TwizzleHandles.handles(coord,color,norm);

    if(this->Mode==0)
      this->MandleData->Surface =
                extractSurface(this->MandleData->Volume,this->Iteration);
    else
      this->MandleData->Surface =
                extractSlice(this->MandleData->Volume,this->Iteration);

    bindSurface(this->MandleData->Surface, coord, color, norm );

    this->Remesh = false;
    }
}

//-----------------------------------------------------------------------------
//called after opengl is inited
void Window::PostInit()
{
  //init the gl handles that the twizzler holds
  this->TwizzleHandles.initHandles();

  //build up our shaders
  this->ShaderProgram.add_vert_shader(make_vertex_shader_code());
  this->ShaderProgram.add_frag_shader(make_fragment_shader_code());
  this->ShaderProgram.build();

  //connect the current time info the the shaders
  //so we can do a moving light source
  this->ShaderLightLocation =
        glGetUniformLocation(this->ShaderProgram.program_id(), "light_xpos");


  glEnable(GL_DEPTH);
  glEnable(GL_DEPTH_TEST);

  //clear the render window
  glClearColor(1.0, 1.0, 1.0, 1.0);

  //compute the mandlebulb
  this->MandleData->Volume =  computeMandlebulb(
                                  dax::make_Vector3(-1,-1,-1),
                                  dax::make_Vector3(0.01,0.01,0.02),
                                  dax::Extent3( dax::make_Id3(0,0,0),
                                                dax::make_Id3(350,350,200) )
                                  );

  this->Remesh = true;
  this->RemeshSurface();
}

//-----------------------------------------------------------------------------
void Window::Display()
{
  this->RemeshSurface();

  GLuint coord, color, norm;
  this->TwizzleHandles.handles(coord,color,norm);

  // Clear Color and Depth Buffers
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  //bind the shaders
  glUseProgram(this->ShaderProgram.program_id());

  //transform the time into an x position to upload to shaders
  //rotate around the model every 7200 seconds (20 * 360).
  float x_pos = (fmod(this->CurrentTime, 20 * 360)) / 20;
  x_pos = 2.5f * dax::math::Cos( dax::math::Pi()  * (x_pos / 180.0f));
  glUniform1f(this->ShaderLightLocation, x_pos );

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

  glDrawArrays( GL_TRIANGLES, 0,
                this->MandleData->Surface.Data.GetNumberOfPoints() );

  glDisableClientState(GL_NORMAL_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  //unbind the shaders
  glUseProgram(0);

  this->DisplayCurrentFPS();

  glutSwapBuffers();
}

//-----------------------------------------------------------------------------
void Window::Idle()
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

  //exit before we try to play the next step
  this->CheckMaxTime();
  this->NextAutoPlayStep();

  glutPostRedisplay();
}

//-----------------------------------------------------------------------------
void Window::ChangeSize(int w, int h)
{
  h = std::max(h,1);
  float ratio =  w * 1.0 / h;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glViewport(0, 0, w, h);
  gluPerspective(45.0f, ratio, 0.01f, 10.0f);
  glMatrixMode(GL_MODELVIEW);
}

//-----------------------------------------------------------------------------
void Window::Key(unsigned char key, int daxNotUsed(x), int daxNotUsed(y) )
{
 if ((key == 27) //escape pressed
     || (key == 'q'))
  {
  exit(0);
  }
}

//-----------------------------------------------------------------------------
void Window::SpecialKey(int key, int daxNotUsed(x), int daxNotUsed(y) )
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

//-----------------------------------------------------------------------------
void Window::Mouse(int button, int state, int x, int y )
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

void Window::MouseMove(int x, int y )
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


//-----------------------------------------------------------------------------
void Window::PassiveMouseMove(int daxNotUsed(x),int daxNotUsed(y))
{

}

} //namespace mandlebulb

