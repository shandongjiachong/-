#include <iostream>
#include <windows.h>
using namespace std;
const int n = 1000, times = 1000;
double sum[n], a[n], b[n][n];
void commonway()
{
    for (int i = 0; i < n; i++)
    {
        sum[i] = 0.0;
        for (int j = 0; j < n; j++)
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
    long long head1, tail1, freq1;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq1);
    QueryPerformanceCounter((LARGE_INTEGER*)&head1);
    for (int i = 1; i <= times; i++)commonway();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
    cout << "Col1: " << (tail1 - head1) * 1000.0 / (freq1 * times) << "ms" << endl;
    return 0;
}