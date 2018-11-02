__kernel void vector_saxpy(__global const int *A, __global const int *B, __global int *C, int a) {

    int i = get_global_id(0);

    // Do the operation
    C[i] = a * A[i] + B[i];
}
