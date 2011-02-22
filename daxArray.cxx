/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxArray.h"

//-----------------------------------------------------------------------------
daxArray::daxArray()
{
  this->Rank = 0;
  this->Shape = NULL;
}

//-----------------------------------------------------------------------------
daxArray::~daxArray()
{
  delete [] this->Shape;
  this->Shape = NULL;
}

//-----------------------------------------------------------------------------
void daxArray::SetShape(int* shape)
{
  delete [] this->Shape;
  this->Shape = NULL;
  if (this->Rank > 0)
    {
    this->Shape = new int[this->Rank];
    for (int cc=0; cc < this->Rank; cc++)
      {
      this->Shape[cc] = shape[cc];
      }
    }
}
