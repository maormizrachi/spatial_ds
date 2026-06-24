#include <array>

template<typename T, int N>
int removeFromArray(std::array<T, N> &array, int index)
{
    if(index >= N or index < 0)
    {
        return;
    }
    for(int i = index; i < N - 1; i++)
    {
        array[i] = array[i + 1];
    }
}
