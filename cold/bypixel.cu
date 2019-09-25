// -*- c++ -*-  ish
__global__ void bypixel(float* out, float* q, float* p, float* t, float* dp, float* dt, int* ndepo)
{
    int row = blockIdx.x;
    int col = blockIdx.y;
    int index = col + row * {{ NTICKS }};
    float pi = row * {{ dPITCH }} + {{ PITCH0 }};
    float ti = col * {{ dTIME }} + {{ TIME0 }};
    float pixel = 0.0;
    float prel=0, trel=0, pnorm=0, tnorm=0, tmp=0;

    // printf("col=%d row=%d index=%d\n", col, row, index);

    for (int idepo=0; idepo < *ndepo; ++idepo) {
        if (q[idepo] == 0.0) {
            continue;
        }
        if (dp[idepo] == 0.0) {
            continue;
        }
        if (dt[idepo] == 0.0) {
            continue;
        }
        prel = pi-p[idepo];
        if (abs(prel) > dp[idepo] * {{ NSIGMA }}) {
            continue;
        }
        trel = ti-t[idepo];
        if (abs(trel) > dt[idepo] * {{ NSIGMA }}) {
            continue;
        }
        pnorm = prel/dp[idepo];
        tnorm = trel/dt[idepo];
        tmp = 0.25 * q[idepo] * exp(-0.5*pnorm*pnorm) * exp(-0.5*tnorm*tnorm) / (dp[idepo]*dt[idepo]);
        pixel += tmp;
    }
    out[index] = pixel;
}
