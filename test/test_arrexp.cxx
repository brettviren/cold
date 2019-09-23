#include <chrono>
#include <random>
#include <iostream>
float gausser(float m, float s)
{ 
    if (s == 0.0) { return 0.0; }
    const float c = 1.0/s;
    const float r = m*s;
    return c * exp(-0.5*r*r);
}
void loopit(int n, float* m, float* s, float* ret)
{
    for (int i=0; i<n; ++i) {
        ret[i] = gausser(m[i], s[i]);
    }
}
void doit(int n, float* m, float* s, float* ret)
{
    for (int i=0; i<n; ++i) {
        if (s[i] == 0.0) { continue; }
        const float c = 1.0/s[i];
        const float r = m[i]*s[i];
        ret[i] = c * exp(-0.5*r*r);
    }
}


int main()
{
    using namespace std::chrono;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.00001,1.0);

    const int nx = 1000;
    const int ny = 1000;
    const int n = nx*ny;
    float* m = new float[n];
    float* s = new float[n];
    float* ret = new float[n];

    for (int i=0; i<n; ++i) {
        s[i] = distribution(generator);
        m[i] = distribution(generator);
    }

    steady_clock::time_point t1 = steady_clock::now();
    //loopit(nx*ny, m, s, ret);
    doit(nx*ny, m, s, ret);
    steady_clock::time_point t2 = steady_clock::now();

    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "cpp speed[kHz]: "
              << n/time_span.count()/1000.0
              << ", time[us]: "
              << time_span.count()*1000.0 << "\n";
}
