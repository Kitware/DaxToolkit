/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include <assert.h>

//-----------------------------------------------------------------------------
template <class T>
daxRegularArray<T>::daxRegularArray()
{
  this->Origin = NULL;
  this->Delta = NULL;
  this->NumberOfItems = 0;
}

//-----------------------------------------------------------------------------
template <class T>
daxRegularArray<T>::~daxRegularArray()
{
  delete [] this->Origin;
  delete [] this->Delta;
}

//-----------------------------------------------------------------------------
template <class T>
void daxRegularArray<T>::SetOrigin(const T* origin)
{
  assert(this->GetRank()==0 || this->GetRank() == 1);
  delete [] this->Origin;
  this->Origin = NULL;
  int num_dims = 0;
  switch (this->GetRank())
    {
  case 0:
    num_dims = 1;
    break;

  case 1:
    num_dims = this->GetShape()[0];
    break;

  default:
    abort();
    }
  this->Origin = new T[num_dims];
  for (int cc=0; cc < num_dims; cc++)
    {
    this->Origin[cc] = origin[cc];
    }
}

//-----------------------------------------------------------------------------
template <class T>
void daxRegularArray<T>::SetDelta(const T* delta)
{
  assert(this->GetRank()==0 || this->GetRank() == 1);
  delete [] this->Delta;
  this->Delta = NULL;
  int num_dims = 0;
  switch (this->GetRank())
    {
  case 0:
    num_dims = 1;
    break;

  case 1:
    num_dims = this->GetShape()[0];
    break;

  default:
    abort();
    }
  this->Delta = new T[num_dims];
  for (int cc=0; cc < num_dims; cc++)
    {
    this->Delta[cc] = delta[cc];
    }
}
