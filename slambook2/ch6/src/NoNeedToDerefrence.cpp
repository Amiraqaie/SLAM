#include <iostream>
#include <vector>

using namespace std;

void print_vector(vector<int> *vec)
{
    for (int i = 0; i < vec->size(); i++)
        cout << vec[0][i] << endl;
}

int main() {
    vector<int> int_vec = {11, 23, 45, 89};
    print_vector(&int_vec);
}