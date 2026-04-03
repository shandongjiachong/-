#include <iostream>
#include <windows.h>
using namespace std;
const int n = 1000,times=10;
double sum[n], a[n], b[n][n];
void advancedway()
{
    for (int i = 0; i < n; i++)
    {
        sum[i] = 0.0;
    }
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            sum[i] += b[j][i] * a[j];
        }
    }
}
int main() {
    for (int i = 0; i < n; i++)a[i] = n - i;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            b[i][j] = i + j;
        }
    }
    long long head2, tail2, freq2;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq2);
    QueryPerformanceCounter((LARGE_INTEGER*)&head2);
    for(int i=1;i<=times;i++)advancedway();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
    cout << "Col2: " << (tail2 - head2) * 1000.0 / (freq2*times) << "ms" << endl;
    return 0;
}