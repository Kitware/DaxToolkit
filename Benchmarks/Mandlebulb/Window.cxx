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
  void release()
  {
    Volume.EscapeIteration.ReleaseResources();

    Surface.Data.GetPointCoordinates().ReleaseResources();
    Surface.Data.GetCellConnections().ReleaseResources();

    Surface.Colors.ReleaseResources();
    Surface.Norms.ReleaseResources();
  }

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
  this->RotateY = 180;
  this->TranslateZ = -3;

  this->Iteration = 1;
  this->ClipRatio = 0.3;
  this->Remesh = true;
  this->Mode = 0;
}

//-----------------------------------------------------------------------------
Window::~Window()
{
  this->Cleanup();

}

//-----------------------------------------------------------------------------
void Window::Cleanup()
{
  //cleanup memory allocation this way so we can handle exit and signal
  //kills and
  this->TwizzleHandles.release();
  this->MandleData->release();
  delete this->MandleData;
}

//-----------------------------------------------------------------------------
void Window::CheckMaxTime()
{
  //terminate once we have been running for ~10 secs
  if((this->MaxTime > 0) && (this->CurrentTime >= 10000))
    {
    this->Cleanup();
    exit(0);
    }
}

//-----------------------------------------------------------------------------
void Window::NextAutoPlayStep()
{
  if(this->AutoPlay)
    {

    const int num_demo_steps = 4;
    int demo_step = (fmod(this->CurrentTime,(60000.0f * num_demo_steps))) / 60000.0f;

    //determine if we do marching cubes or clip based on the time.
    //we cycle through clip / marching cubes every 1 minute
    //so demo step 0,2 = mc, 1,3 = clip
    this->Mode  = (fmod(this->CurrentTime,120000.0f) < 60000) ?  0 : 1;

    //move the clip plane and not iso value second 4
    //move the clip plane and iso value for second 2
    if(demo_step <= 2)
      {
      //alternating every 60 seconds move iteration from 1 to 30, and back to 1
      int sign = (fmod(this->CurrentTime,60000.0f) < 30000) ?  1 : -1;
      this->Iteration = dax::math::Min(30.0f, this->Iteration + (sign * 0.2f));
      this->Iteration = dax::math::Max(1.0f, this->Iteration);
      }
    if(demo_step == 1 || demo_step == 3)
      {
      int sign = (fmod(this->CurrentTime,60000.0f) < 30000) ?  1 : -1;
      this->ClipRatio = dax::math::Max(0.3f, this->ClipRatio + (sign * 0.02f) );
      this->ClipRatio = dax::math::Min(1.0f, this->ClipRatio);
      }
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
                extractCut(this->MandleData->Volume,
                             this->ClipRatio,this->Iteration);

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
  this->Shaders.add_vert_shader(make_vertex_shader_code());
  this->Shaders.add_frag_shader(make_fragment_shader_code());
  this->Shaders.build();

  //connect the current time info the the shaders
  //so we can do a moving light source
  this->ShaderLightLocation =
        glGetUniformLocation(this->Shaders.program_id(), "light_xpos");


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
  glUseProgram(this->Shaders.program_id());

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

  glBindBuffer(GL_ARRAY_BUFFER, 0 );

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
  this->Cleanup();
  exit(0);
  }

  if(key == 32 ) //space bar pressed
    {
    //flip the mode we are using
    this->Mode = !this->Mode;
    this->Remesh = true;
    }
}

//-----------------------------------------------------------------------------
void Window::SpecialKey(int key, int daxNotUsed(x), int daxNotUsed(y) )
{
  switch (key)
    {
    case GLUT_KEY_UP:
      this->Iteration = dax::math::Min(35.0, this->Iteration + 0.05);
      this->Remesh = true;
      break;
    case GLUT_KEY_DOWN :
      this->Iteration = dax::math::Max(0.0, this->Iteration - 0.05);
      this->Remesh = true;
      break;
    case GLUT_KEY_LEFT:
      this->ClipRatio = dax::math::Max(0.3, this->ClipRatio - 0.01);
      this->Remesh = true;
      break;
    case GLUT_KEY_RIGHT:
      this->ClipRatio = dax::math::Min(1.0, this->ClipRatio + 0.01);
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

