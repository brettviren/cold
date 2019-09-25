// -*- c++ -*-  ish
__global__ void bypixel(float* field, float* bindesc, int* offset, float* depo)
{
    float t0 = bindesc[0];
    float dt = bindesc[1];
    float p0 = bindesc[2];
    float dp = bindesc[3];

    float ampli = depo[0];
    float tmean = depo[1];
    float tsig  = depo[2];
    float pmean = depo[3];
    float psig  = depo[4];

    int col = threadIdx.x;
    int row = threadIdx.y;

    float t = t0 + dt * col;
    float p = p0 + dp * row;

    float trel = t - tmean;
    float prel = p - pmean;
    
    float pnorm = prel/psig;
    float tnorm = trel/tsig;
    float tmp = 0.25 * ampli * exp(-0.5*pnorm*pnorm) * exp(-0.5*tnorm*tnorm) / (dp*dt);
    int oindex = offset[0]+col + (offset[1]+row)*offset[2];
    field[oindex] += tmp;
}
