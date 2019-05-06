// Microsoft (c) 2019, Wenxiang Hu
#include "cuda_to_rocm_pass.hpp"
#include "../../cuda/cuda_langunit.hpp"
#include "../rocm_langunit.hpp"

using namespace nnfusion::rocm;

bool CudatoROCM::run(ir::Operator_p& inter_op)
{
    auto cop = static_pointer_cast<ir::Function>(inter_op);
    // Assume all dependency are merged into cop->dep_unit
    return convert(cop->dep_unit);
}

bool CudatoROCM::convert(LanguageUnit_p p)
{
    p->remove(cuda::header::cuda);
    p->remove(cuda::header::cudnn);
    p->remove(cuda::header::cublas);
    p->replace(cuda::declaration::division_by_invariant_multiplication,
               rocm::declaration::division_by_invariant_multiplication);
    p->replace(cuda::declaration::load, rocm::declaration::load);
    p->require(rocm::header::nnfusion_hip);
    return true;
}