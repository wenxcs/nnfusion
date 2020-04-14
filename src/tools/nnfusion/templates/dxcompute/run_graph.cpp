#include "d3dx12_nnfusion.h"

int main(int argc, char** argv)
{
    D3DDevice device(false, false);
    device.Init();

    using namespace nnfusion_dml;

    std::vector<ID3D12CommandList*> cmdQueue;

#include "nnfusion_rt.h"

    system("pause");
    return 0;
}
