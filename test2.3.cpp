#include <iostream>
#include <windows.h>
using namespace std;
const int n = 16, times = 1000;
double a[n];
void way()
{
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i];
        s1 += a[i + 1];
        s2 += a[i + 2];
        s3 += a[i + 3];
    }
    double sum = s0 + s1 + s2 + s3;
    for (; i < n; ++i) sum += a[i];
}

int main()
{
    for (int i = 0; i < n; i++)a[i] = i + 1;
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 1; i <= times; i++)way();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "Col: " << (tail - head) * 1000.0 / (freq * times) << "ms" << endl;
    return 0;
}