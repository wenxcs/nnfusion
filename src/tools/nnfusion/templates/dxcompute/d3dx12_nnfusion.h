#pragma once

#include "d3dx12_helper.h"

namespace nnfusion_dml {

	template<class T>
	std::vector<T> load_data(const std::string &name, size_t num_elements) {
		std::vector<T> ret(num_elements);

		std::ifstream t("Constant\\" + name);
		if (t.fail()) {
		  if (name != "")
			  std::cout << "[Warn] Cannot find constant data from: `Constant\\" << name << "`, going to fill with pre-defined values." << std::endl;
			std::fill(ret.begin(), ret.end(), 1);
		}
		else {
			std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
			assert(str.size() == num_elements * sizeof(T));
			memcpy(ret.data(), str.data(), str.size());
		}
		return std::move(ret);
	}

	class NNfusionTensor {
		ComPtr<ID3D12Resource> deviceGPUSrcX;
		std::vector<size_t> shape;
		size_t type_size;

	public:
		NNfusionTensor(D3DDevice& device, const std::vector<size_t>& shape, size_t type_size): shape(shape), type_size(type_size) {
			device.CreateGPUOnlyResource(type_size * NumElements(), &deviceGPUSrcX);
		}

		size_t NumElements() const {
			return std::accumulate(
				shape.begin(), shape.end(), 1LU, std::multiplies<size_t>());
		}

		size_t TypeSize() const {
			return type_size;
		}

		ComPtr<ID3D12Resource> Data() const {
			return deviceGPUSrcX;
		}

		std::vector<size_t> Shape() const {
			return shape;
		}
	};

	class NNfusionMemcpy {
		ComPtr<ID3D12Resource> deviceGPUSrcX;
		ComPtr<ID3D12Resource> deviceCPUSrcX;
		ComPtr<ID3D12CommandAllocator> pCommandAllocator;
		ComPtr<ID3D12GraphicsCommandList> m_computeCommandList;
		size_t bufferSize;

	public:
		NNfusionMemcpy(D3DDevice& device, std::vector<ID3D12CommandList*>& cmdQueue, NNfusionTensor& dst, void *src) {
			bufferSize = dst.TypeSize() * dst.NumElements();

			deviceGPUSrcX = dst.Data();
			device.CreateUploadBuffer(bufferSize, &deviceCPUSrcX);
			device.MapAndCopyToResource(deviceCPUSrcX.Get(), src, bufferSize);

			IFE(device.pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&pCommandAllocator)));
			IFE(device.pDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, pCommandAllocator.Get(), nullptr, IID_PPV_ARGS(&m_computeCommandList)));
			m_computeCommandList->CopyResource(deviceGPUSrcX.Get(), deviceCPUSrcX.Get());
			m_computeCommandList->Close();

			cmdQueue.push_back(Launch());
		}

		NNfusionMemcpy(D3DDevice& device, std::vector<ID3D12CommandList*>& cmdQueue, void *dst, NNfusionTensor& src) {
			bufferSize = src.TypeSize() * src.NumElements();

			deviceGPUSrcX = src.Data();
			device.CreateReadbackBuffer(bufferSize, &deviceCPUSrcX);

			IFE(device.pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&pCommandAllocator)));
			IFE(device.pDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, pCommandAllocator.Get(), nullptr, IID_PPV_ARGS(&m_computeCommandList)));
			m_computeCommandList->CopyResource(deviceCPUSrcX.Get(), deviceGPUSrcX.Get());
			m_computeCommandList->Close();

			cmdQueue.push_back(Launch());
		}

		ID3D12GraphicsCommandList* Launch() {
			return m_computeCommandList.Get();
		}

		template <class T>
		void PrintStageBuffer(D3DDevice& device, const std::string &name)
		{
			assert(bufferSize % sizeof(T) == 0);
			std::vector<T> dst(bufferSize / sizeof(T));
			device.MapCopyFromResource(deviceCPUSrcX.Get(), dst.data(), bufferSize);
			T* buffer = (T*)dst.data();
			std::cout << "Result(" << name << ") = {";
			for (int i = 0; i < dst.size(); ++i) {
				if (i)
					std::cout << ", ";
				std::cout << dst[i];
			}
			std::cout << "}\n" << std::endl;
		}
	};

	class NNfusionOperator {
		ComPtr<ID3D12GraphicsCommandList> m_computeCommandList;

		ComPtr<ID3D12RootSignature> m_computeRootSignature;
		ComPtr<ID3DBlob> computeShader;
		ComPtr<ID3D12PipelineState> m_computeState;
		ComPtr<ID3D12CommandAllocator> computeCommandAllocator;
		D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc;

		LPCWSTR hlsl_source;
	public:
		NNfusionOperator(D3DDevice& device, std::vector<ID3D12CommandList*>& cmdQueue, const std::vector<NNfusionTensor>& inputs, const std::vector<NNfusionTensor>& outputs, const std::vector<UINT> &threads, LPCWSTR hlsl_source)
				: hlsl_source(hlsl_source) {

			// Prepare Root
			std::vector<CD3DX12_ROOT_PARAMETER1> computeRootParameters(inputs.size() + outputs.size());
			for (int i = 0; i < inputs.size(); ++i)
				computeRootParameters[i].InitAsShaderResourceView(i);
			for (int i = 0; i < outputs.size(); ++i)
				computeRootParameters[inputs.size() + i].InitAsUnorderedAccessView(i);

			CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC computeRootSignatureDesc;
			computeRootSignatureDesc.Init_1_1(computeRootParameters.size(), computeRootParameters.data());

			ComPtr<ID3DBlob> signature;
			ComPtr<ID3DBlob> error;

			IFE(D3DX12SerializeVersionedRootSignature(&computeRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error));
			IFE(device.pDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_computeRootSignature)));

			auto path = std::wstring(L"HLSL\\") + hlsl_source;
			std::ifstream fin(path);
			if (fin.fail()) {
				std::wcout << L"[Error] Cannot find HLSL data from: `" << path << L"`, please copy the full codegen folder!" << std::endl;
				_exit(1);
			}
			fin.close();
			IFE(D3DCompileFromFile(path.c_str(), NULL, NULL, "CSMain", "cs_5_0", 0, 0, &computeShader, NULL));


			computePsoDesc.pRootSignature = m_computeRootSignature.Get();
			computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());

			IFE(device.pDevice->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&m_computeState)));
			IFE(device.pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&computeCommandAllocator)));
			IFE(device.pDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, computeCommandAllocator.Get(), m_computeState.Get(), IID_PPV_ARGS(&m_computeCommandList)));

			m_computeCommandList->SetComputeRootSignature(m_computeRootSignature.Get());
			for (int i = 0; i < inputs.size(); ++i)
				m_computeCommandList->SetComputeRootShaderResourceView(i, inputs[i].Data()->GetGPUVirtualAddress());
			for (int i = 0; i < outputs.size(); ++i)
				m_computeCommandList->SetComputeRootUnorderedAccessView(inputs.size() + i, outputs[i].Data()->GetGPUVirtualAddress());
			m_computeCommandList->Dispatch(threads[0], threads[1], threads[2]);
			IFE(m_computeCommandList->Close());

			cmdQueue.push_back(Launch());
		}

		ID3D12GraphicsCommandList* Launch() {
			return m_computeCommandList.Get();
		}
	};
}