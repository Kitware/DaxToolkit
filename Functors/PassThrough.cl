/*=========================================================
  Simple PassThrough functor for illustration.
=========================================================*/
// FUNCTOR MODULE VERSION 0.0.1
// MODULE PassThrough
// INPUT inarray;any_array;none;
// OUTPUT outarray;any_array;
void PassThrough(const float *inarray, float *outarray)
{
  *(outarray) = *(inarray);
}
