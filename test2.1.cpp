#include <iostream>
#include <windows.h>
using namespace std;
const int n = 16, times = 1000;
double a[n];
void commonway()
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)sum += a[i];
    cout << sum;
}

int main()
{
    for (int i = 0; i < n; i++)a[i] = i + 1;
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 1; i <= times; i++)commonway();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "Col: " << (tail - head) * 1000.0 / (freq * times) << "ms" << endl;
    return 0;
}