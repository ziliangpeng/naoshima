#include <stdio.h>

int cfunc_i(int i)
{
    printf("print from cfunc_i with %d\n", i);
    return i;
}

int cfunc()
{
    printf("print from cfunc\n");
    return 0;
}
