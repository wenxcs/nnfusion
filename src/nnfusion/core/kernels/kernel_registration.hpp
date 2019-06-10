// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "common.hpp"
#include "kernel_emitter.hpp"

namespace nnfusion
{
    namespace kernels
    {
        class KernelRegistration
        {
        public:
            typedef shared_ptr<KernelEmitter> (*Factory)(shared_ptr<KernelContext>);

            // Starts with just the name field set. Required
            explicit KernelRegistration(const string op_name);
            ~KernelRegistration(){};

            // Required: specify the device type (e.g., CUDA_GPU) this kernel supports.
            // Return *this
            KernelRegistration& Device(const DeviceType device_type);

            // Specify the data (inputs/outputs) types this kernel supports
            // Return *this
            KernelRegistration& TypeConstraint(const DataType data_type);

            // Add an arbitrary user-defined tag on the kernel to allow the operator
            // to choose this kernel
            // Return *this
            KernelRegistration& Tag(const string tag);

            // Required: specify the kernel factory that creates this kernel emitter
            // Return *this
            KernelRegistration& KernelFactory(const Factory factory);

            // The final step to create an kernel emitter registration
            const shared_ptr<KernelRegistration> Build();

            const string op_name() { return m_op_name; }
            void debug_string() const
            {
                std::cout << "m_op_name: " << m_op_name << "\n"
                          << "m_device_type: " << m_device_type << "\n"
                          << "m_data_type: " << m_data_type << "\n"
                          << "m_tag: " << m_tag << "\n"
                          << "m_factory: " << m_factory << std::endl;
            }

        private:
            friend class KernelRegistry;
            string m_op_name;
            DeviceType m_device_type;
            DataType m_data_type;
            string m_tag;
            Factory m_factory;
        };

        static KernelRegistration& Name(const string op_name)
        {
            // TODO(jxue): managed with a shared ptr
            KernelRegistration* registration = new KernelRegistration(op_name);
            return *registration;
        }

        class KernelRegistry
        {
        public:
            KernelRegistry(){};
            ~KernelRegistry(){};
            bool RegisterKernel(const string op_name, shared_ptr<KernelRegistration> registration);
            shared_ptr<const KernelRegistration> FindKernelRegistration(
                const string op_name, const DeviceType& device_type, const DataType data_type);
            shared_ptr<const KernelRegistration>
                KernelSelect(std::vector<shared_ptr<const KernelRegistration>>& matched_regs);

            size_t RegisteredKernelSize() const { return m_kernel_registry.size(); }
            static KernelRegistry* Global()
            {
                static KernelRegistry* global_kernel_registry = new KernelRegistry();
                return global_kernel_registry;
            }

        private:
            std::unordered_multimap<string, shared_ptr<KernelRegistration>> m_kernel_registry;
        };

        class KernelRegistrar
        {
        public:
            KernelRegistrar(const string op_name, shared_ptr<KernelRegistration> registration)
            {
                std::cout << "KernelRegistrar constructor" << std::endl;
                KernelRegistry::Global()->RegisterKernel(op_name, registration);
            }
        };

#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) CONCAT_IMPL(x, y)

#define REGISTER_KERNEL_EMITTER(op_name, attrs, constructor)                                       \
    KernelRegistrar CONCAT(kernel_registrar, __COUNTER__)(                                         \
        op_name,                                                                                   \
        Name(op_name)                                                                              \
            .attrs                                                                                 \
            .KernelFactory([](shared_ptr<KernelContext> context) -> shared_ptr<KernelEmitter> {    \
                return make_shared<cuda::Pad>(context);                                            \
            })                                                                                     \
            .Build());

    } // namespace kernels
} // namespace nnfusion