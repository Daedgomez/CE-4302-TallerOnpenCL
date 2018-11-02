__kernel void matrix_mul(__global const int * A, __global const int * B, __global int * C)
{
   int i = get_global_id(0);
   int j = get_global_id(1);

   int res = 0;
   for (int k = 0; k < 4; ++k)
   {
      res += A[j * 4 + k] * B[k * 4 + i];
   }
   C[j * 4 + i] = res;
}
