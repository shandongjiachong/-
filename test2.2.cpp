#include <iostream>
#include <windows.h>
using namespace std;
const int n = 16, times = 1000;
double a[n];
void advancedway(const double *a) { 
    int n1 = n;
    double temp[n];
    for (int i = 0; i < n; ++i) temp[i] = a[i];
    while (n1 > 1) {
        for (int i = 0; i < n1 / 2; ++i) {
            temp[i] = temp[2 * i] + temp[2 * i + 1];
        }
        n1 = (n1 + 1) / 2;  
    }
    double ans=temp[0];
}
int main()
{
    for (int i = 0; i < n; i++)a[i] = i + 1;
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 1; i <= times; i++)advancedway(a);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "Col: " << (tail - head) * 1000.0 / (freq * times) << "ms" << endl;
    return 0;
}
