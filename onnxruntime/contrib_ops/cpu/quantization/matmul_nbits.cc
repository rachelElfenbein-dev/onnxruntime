// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/quantization/matmul_nbits_impl.h"

#include <cstdint>
#include <type_traits>

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/common.h"
#include "contrib_ops/cpu/quantization/matmul_nbits_helper.h"

namespace onnxruntime {
namespace contrib {

namespace {

// MatMulNBits op input indices.
// These should match the inputs names specified in the op schema.
namespace InputIndex {
constexpr size_t A = 0,
                 B = 1,
                 scales = 2,
                 zero_points = 3,
                 g_idx = 4,
                 bias = 5;
};

typedef enum {
  Level0, /*!< input fp32, accumulator fp32 */
  Level1, /*!< input fp32, accumulator fp32 */
  Level2, /*!< input fp16, accumulator fp16 */
  Level3, /*!< input bf16, accumulator fp32 */
  Level4, /*!< input int8, accumulator int32 */
} ACCURACY_LEVEL;

// T: A data type.
template <typename T>
MLAS_QNBIT_GEMM_COMPUTE_TYPE
GetComputeType(size_t nbits, size_t block_size, int64_t accuracy_level_attr) {
  // For Fp32, only accuracy level 1 or 4 makes sense.
  // non-ARM CPU converts Fp16 to Fp32.
  // By converting Fp32 to Fp16, precision becomes worse. And due to the casting,
  // there is no performance gain.
  if (accuracy_level_attr == static_cast<int64_t>(Level4) &&
      MlasIsQNBitGemmAvailable(nbits, block_size, SQNBIT_CompInt8)) {
    return SQNBIT_CompInt8;
  }

  return SQNBIT_CompFp32;
}

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
template <>
MLAS_QNBIT_GEMM_COMPUTE_TYPE
GetComputeType<MLFloat16>(size_t nbits, size_t block_size, int64_t accuracy_level_attr) {
  // For Fp16, only accuracy level 2 or 4 makes sense.
  // By converting Fp16 to Fp32, there is not precision increase, and the performance
  // becomes worse.
  if (accuracy_level_attr == static_cast<int64_t>(Level4) &&
      MlasIsQNBitGemmAvailable(nbits, block_size, HQNBIT_CompInt8)) {
    return HQNBIT_CompInt8;
  }

  // if HQNBIT_CompFp16 is not supported, will fallback to unpacked computation.
  return HQNBIT_CompFp16;
}
#endif  // !MLAS_F16VEC_INTRINSICS_SUPPORTED || !MLAS_TARGET_ARM64

}  // namespace

bool GetType(const NodeArg& node_arg, int32_t& type) {
  type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  const auto* type_proto = node_arg.TypeAsProto();
  if (!type_proto || !type_proto->has_tensor_type() || !type_proto->tensor_type().has_elem_type()) {
    return false;
  }

  type = type_proto->tensor_type().elem_type();
  return true;
}

// T1 is the type of the input matrix A, scales and biases.
// Use class level template to facilitate specialization for different types.
template <typename T1>
class MatMulNBits final : public OpKernel {
 public:
  MatMulNBits(const OpKernelInfo& info)
      : OpKernel(info),
        K_{narrow<size_t>(info.GetAttr<int64_t>("K"))},
        N_{narrow<size_t>(info.GetAttr<int64_t>("N"))},
        block_size_{narrow<size_t>(info.GetAttr<int64_t>("block_size"))},
        nbits_{narrow<size_t>(info.GetAttr<int64_t>("bits"))},
        has_g_idx_{info.GetInputCount() > InputIndex::g_idx && info.node().InputDefs()[InputIndex::g_idx]->Exists()},
        has_bias_{info.GetInputCount() > InputIndex::bias && info.node().InputDefs()[InputIndex::bias]->Exists()},
        compute_type_{GetComputeType<T1>(nbits_, block_size_, info.GetAttr<int64_t>("accuracy_level"))} {
    const auto& node = info.node();
    auto input_defs = node.InputDefs();
    const NodeArg* zero_point_arg =
        (info.GetInputCount() > InputIndex::zero_points && input_defs[InputIndex::zero_points]->Exists())
            ? input_defs[3]
            : nullptr;

    if (int32_t type; zero_point_arg && GetType(*zero_point_arg, type)) {
      has_unquantized_zero_point_ = type != ONNX_NAMESPACE::TensorProto_DataType_UINT8;
    }

    ORT_ENFORCE(nbits_ == 2 || nbits_ == 4 || nbits_ == 8,
                "Only 2b, 4b and 8b quantization is supported for MatMulNBits op, additional bits support is planned.");
    const Tensor* tensor_zero_point = nullptr;
    has_zp_input_ = info.TryGetConstantInput(InputIndex::zero_points, &tensor_zero_point);
  }

  Status Compute(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers, int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

 private:
  const size_t K_;
  const size_t N_;
  const size_t block_size_;
  const size_t nbits_;
  const bool has_g_idx_;
  const bool has_bias_;
  bool scales_are_packed_{false};
  const MLAS_QNBIT_GEMM_COMPUTE_TYPE compute_type_;
  bool has_unquantized_zero_point_{false};
  const bool column_wise_quant_{true};
  IAllocatorUniquePtr<void> packed_b_{};
  size_t packed_b_size_{0};
  IAllocatorUniquePtr<float> scales_fp32_{};
  IAllocatorUniquePtr<float> bias_fp32_{};

  // NEW: 3D
  std::vector<BufferUniquePtr> packed_b_batches_;  // גודל B
  size_t packed_b_size_per_batch_{0};
  bool has_multi_packed_b_{false};

  bool has_zp_input_{false};

  // dequantize B first and then compute float gemm
  Status ComputeBUnpacked(const Tensor* a,
                          const Tensor* b,
                          const Tensor* scales,
                          const Tensor* zero_points,
                          const Tensor* reorder_idx,
                          const Tensor* bias,
                          Tensor* y,
                          AllocatorPtr& allocator,
                          concurrency::ThreadPool* thread_pool,
                          const MatMulComputeHelper& helper) const {
    ORT_THROW("ComputeBUnpacked is not supported for T1 type.");
  }

  Status ComputeBPacked(const Tensor* a,
                        const Tensor* scales,
                        const Tensor* zero_points,
                        const Tensor* bias,
                        Tensor* y,
                        AllocatorPtr& allocator,
                        concurrency::ThreadPool* thread_pool,
                        const MatMulComputeHelper& helper) const;

  Status ComputeBUnpacked3D(const Tensor* a,
                            const Tensor* b,
                            const Tensor* scales,
                            const Tensor* zero_points,
                            const Tensor* reorder_idx,
                            const Tensor* bias,
                            Tensor* y,
                            AllocatorPtr& allocator,
                            concurrency::ThreadPool* thread_pool,
                            const MatMulComputeHelper& helper,
                            int64_t B_nbits) const;
};


template <typename T1>
Status MatMulNBits<T1>::PrePack(const Tensor& tensor,
                                int input_idx,
                                /*out*/ AllocatorPtr alloc,
                                /*out*/ bool& is_packed,
                                /*out*/ PrePackedWeights* prepacked_weights) {
  ORT_UNUSED_PARAMETER(prepacked_weights);
  is_packed = false;

  if (has_g_idx_ || has_unquantized_zero_point_) {
    return Status::OK();
  }

  {
    const TensorShape& s = tensor.Shape();
    const int rank = s.NumDimensions();

    if (input_idx == InputIndex::B && rank == 4) {
      return Status::OK();
    }

    if (input_idx == InputIndex::scales && rank == 3 && s[0] > 1) {
      return Status::OK();
    }

    if (input_idx == InputIndex::zero_points && rank == 3 && s[0] > 1) {
      return Status::OK();
    }
  }

  if (!MlasIsQNBitGemmAvailable(nbits_, block_size_, compute_type_)) {
    return Status::OK();
  }

  if (input_idx == InputIndex::B) {
    const Tensor* scales = nullptr;
    OpKernel::Info().TryGetConstantInput(InputIndex::scales, &scales);

    packed_b_size_ = MlasQNBitGemmPackQuantBDataSize(N_, K_, nbits_, block_size_, has_zp_input_, compute_type_);
    if (packed_b_size_ == 0) {
      return Status::OK();
    }

    const void* qptr = tensor.DataRaw();
    const void* sptr = scales ? scales->DataRaw() : nullptr;

    packed_b_ = IAllocator::MakeUniquePtr<void>(alloc, packed_b_size_, true);
    MlasQNBitGemmPackQuantBData(
        N_, K_, nbits_, block_size_, compute_type_,
        qptr, /* packed_dst */ packed_b_.get(),
        sptr,          /* scales      */
        has_zp_input_, /* has zp      */
        nullptr,       /* zp          */
        nullptr);      /* reorder idx */
    is_packed = true;

  } else if (compute_type_ == SQNBIT_CompInt8) {
    bool should_pack_scale_and_zp_inputs = [&]() {
#if defined(MLAS_TARGET_AMD64_IX86)
      return true;
#else
      return (nbits_ == 8);
#endif
    }();

    if (should_pack_scale_and_zp_inputs) {
      if (input_idx == InputIndex::scales && packed_b_ != nullptr) {
        auto sptr = tensor.Data<float>();
        MlasQNBitGemmPackQuantBData(N_, K_, nbits_, block_size_, compute_type_,
                                    nullptr, packed_b_.get(), sptr,
                                    has_zp_input_, nullptr, nullptr);
        is_packed = false;
      }

      if (input_idx == InputIndex::zero_points && packed_b_ != nullptr) {
        auto zptr = tensor.Data<uint8_t>();
        MlasQNBitGemmPackQuantBData(N_, K_, nbits_, block_size_, compute_type_,
                                    nullptr, packed_b_.get(), nullptr,
                                    has_zp_input_, zptr, nullptr);
        is_packed = false;
      }
    }

#if defined(MLAS_TARGET_ARM64)
    if (input_idx == InputIndex::scales &&
        packed_b_ != nullptr &&
        MlasQNBitGemmScalesPacked(K_, nbits_, block_size_, compute_type_, has_zp_input_)) {
      // שימי לב: זה יקרה רק במסלול 2D; ב-3D כבר חזרנו קודם.
      scales_are_packed_ = true;
      is_packed = true;
    }
#endif  // MLAS_TARGET_ARM64
  }

  return Status::OK();
}

#if !defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) || !defined(MLAS_TARGET_ARM64)
// Non-ARM-with-fp16-intrinsics fall back fp16 to fp32.
template <>
Status MatMulNBits<MLFloat16>::PrePack(const Tensor& tensor,
                                       int input_idx,
                                       /*out*/ AllocatorPtr alloc,
                                       /*out*/ bool& is_packed,
                                       /*out*/ PrePackedWeights* prepacked_weights) {
  ORT_UNUSED_PARAMETER(prepacked_weights);

  if (input_idx == InputIndex::scales || input_idx == InputIndex::bias) {
    auto sptr = tensor.Data<MLFloat16>();
    auto tensor_size = static_cast<size_t>(tensor.Shape().Size());
    auto ptr = IAllocator::MakeUniquePtr<float>(alloc, tensor_size, true);
    MlasConvertHalfToFloatBuffer(sptr, ptr.get(), tensor_size);
    if (input_idx == InputIndex::scales) {
      scales_fp32_ = std::move(ptr);
    } else {
      bias_fp32_ = std::move(ptr);
    }
  }

  is_packed = false;

  if (has_g_idx_ || has_unquantized_zero_point_) {
    return Status::OK();
  }

  {
    const TensorShape& s = tensor.Shape();
    const int rank = s.NumDimensions();

    if (input_idx == InputIndex::B && rank == 4) {
      return Status::OK();
    }
    if (input_idx == InputIndex::scales && rank == 3 && s[0] > 1) {
      return Status::OK();
    }
    if (input_idx == InputIndex::zero_points && rank == 3 && s[0] > 1) {
      return Status::OK();
    }
  }

  if (!MlasIsQNBitGemmAvailable(nbits_, block_size_, compute_type_)) {
    return Status::OK();
  }

  if (input_idx == InputIndex::B) {
    const Tensor* scales = nullptr;
    OpKernel::Info().TryGetConstantInput(InputIndex::scales, &scales);
    if (scales && MlasQNBitGemmScalesPacked(K_, nbits_, block_size_, compute_type_, has_zp_input_)) {
      auto sptr = scales->Data<MLFloat16>();
      auto tensor_size = static_cast<size_t>(tensor.Shape().Size());
      auto ptr = IAllocator::MakeUniquePtr<float>(alloc, tensor_size, true);
      MlasConvertHalfToFloatBuffer(sptr, ptr.get(), tensor_size);
      scales_fp32_ = std::move(ptr);
    }

    packed_b_size_ = MlasQNBitGemmPackQuantBDataSize(N_, K_, nbits_, block_size_, has_zp_input_, compute_type_);
    if (packed_b_size_ == 0) {
      return Status::OK();
    }
    auto qptr = tensor.DataRaw();
    packed_b_ = IAllocator::MakeUniquePtr<void>(alloc, packed_b_size_, true);
    MlasQNBitGemmPackQuantBData(N_, K_, nbits_, block_size_, compute_type_,
                                qptr, packed_b_.get(),
                                scales_fp32_.get(),
                                has_zp_input_, nullptr, nullptr);
    is_packed = true;

  } else if (compute_type_ == SQNBIT_CompInt8) {
#ifdef MLAS_TARGET_AMD64_IX86
    if (input_idx == InputIndex::scales && packed_b_ != nullptr) {
      MlasQNBitGemmPackQuantBData(N_, K_, nbits_, block_size_, compute_type_,
                                  nullptr, packed_b_.get(),
                                  scales_fp32_.get(),
                                  has_zp_input_, nullptr, nullptr);
      is_packed = false;
    } else if (input_idx == InputIndex::zero_points && packed_b_ != nullptr) {
      auto zptr = tensor.Data<uint8_t>();
      MlasQNBitGemmPackQuantBData(N_, K_, nbits_, block_size_, compute_type_,
                                  nullptr, packed_b_.get(),
                                  nullptr, has_zp_input_,
                                  zptr, nullptr);
      is_packed = false;
    }
#endif  // MLAS_TARGET_AMD64_IX86
  }

  return Status::OK();
}
#endif

template <typename T1>
Status MatMulNBits<T1>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers, int input_idx,
                                                  /*out*/ bool& used_shared_buffers) {
  used_shared_buffers = false;

  if (input_idx == 1) {
    used_shared_buffers = true;
    packed_b_ = std::move(prepacked_buffers[0]);
  }

  return Status::OK();
}

template <typename T1>
Status MatMulNBits<T1>::ComputeBPacked(const Tensor* a,
                                       const Tensor* scales,
                                       const Tensor* zero_points,
                                       const Tensor* bias,
                                       Tensor* y,
                                       AllocatorPtr& allocator,
                                       concurrency::ThreadPool* thread_pool,
                                       const MatMulComputeHelper& helper) const {
  const auto* a_data = a->Data<T1>();
  const auto* scales_data = scales == nullptr ? nullptr : scales->Data<T1>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->DataRaw();
  const auto* bias_data = bias == nullptr ? nullptr : bias->Data<T1>();
  auto* y_data = y->MutableData<T1>();

  const size_t batch_count = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(false);

  IAllocatorUniquePtr<std::byte> workspace{};
  const size_t workspace_size = MlasQNBitGemmBatchWorkspaceSize(
      M, N, K, batch_count, nbits_, block_size_, zero_points, compute_type_);
  if (workspace_size > 0) {
    // Use reserve since no caching is needed
    workspace = IAllocator::MakeUniquePtr<std::byte>(allocator, workspace_size, true);
  }

  InlinedVector<MLAS_QNBIT_GEMM_DATA_PARAMS<T1>> data(batch_count);
  for (size_t i = 0; i < batch_count; ++i) {
    data[i].A = a_data + helper.LeftOffsets()[i];
    data[i].lda = lda;
    if (compute_type_ == SQNBIT_CompInt8) {
      data[i].QuantBDataWorkspace = packed_b_.get();
    }
    data[i].PackedQuantBData = static_cast<std::byte*>(packed_b_.get());
    data[i].QuantBScale = scales_data;
    data[i].QuantBZeroPoint = zero_points_data;
    data[i].Bias = bias_data;
    data[i].C = y_data + helper.OutputOffsets()[i];
    data[i].ldc = N;
  }
  MlasQNBitGemmBatch(M, N, K, batch_count, nbits_, block_size_, compute_type_, data.data(), workspace.get(),
                     thread_pool);
  return Status::OK();
}

#if !defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) || !defined(MLAS_TARGET_ARM64)
template <>
Status MatMulNBits<MLFloat16>::ComputeBPacked(const Tensor* a,
                                              const Tensor* scales,
                                              const Tensor* zero_points,
                                              const Tensor* bias,
                                              Tensor* y,
                                              AllocatorPtr& allocator,
                                              concurrency::ThreadPool* thread_pool,
                                              const MatMulComputeHelper& helper) const {
  const auto* a_data = a->Data<MLFloat16>();
  const auto* scales_data = scales->Data<MLFloat16>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->DataRaw();
  const auto* bias_data = bias == nullptr ? nullptr : bias->Data<MLFloat16>();
  auto* y_data = y->MutableData<MLFloat16>();

  const size_t batch_count = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(false);

  IAllocatorUniquePtr<std::byte> workspace{};
  const size_t workspace_size = MlasQNBitGemmBatchWorkspaceSize(
      M, N, K, batch_count, nbits_, block_size_, zero_points, compute_type_);
  if (workspace_size > 0) {
    // Use reserve since no caching is needed
    workspace = IAllocator::MakeUniquePtr<std::byte>(allocator, workspace_size, true);
  }

  auto a_size = static_cast<size_t>(a->Shape().Size());
  auto tmp_a_data_ptr = IAllocator::MakeUniquePtr<float>(allocator, a_size, true);
  MlasConvertHalfToFloatBuffer(a_data, tmp_a_data_ptr.get(), a_size);

  float* scales_ptr = nullptr;
  if (!scales_fp32_) {
    auto scales_temp = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(scales->Shape().Size()), true);
    MlasConvertHalfToFloatBuffer(scales_data, scales_temp.get(), static_cast<size_t>(scales->Shape().Size()));
    scales_ptr = scales_temp.get();
  } else {
    scales_ptr = scales_fp32_.get();
  }

  float* bias_ptr = nullptr;
  if (bias_data) {
    if (!bias_fp32_) {
      auto bias_temp = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(bias->Shape().Size()), true);
      MlasConvertHalfToFloatBuffer(bias_data, bias_temp.get(), static_cast<size_t>(bias->Shape().Size()));
      bias_ptr = bias_temp.get();
    } else {
      bias_ptr = bias_fp32_.get();
    }
  }

  size_t c_size = static_cast<size_t>(y->Shape().Size());
  std::vector<float> c_v(c_size);

  InlinedVector<MLAS_QNBIT_GEMM_DATA_PARAMS<float>> data(batch_count);
  for (size_t i = 0; i < batch_count; ++i) {
    data[i].A = tmp_a_data_ptr.get() + helper.LeftOffsets()[i];
    data[i].lda = lda;
#ifdef MLAS_TARGET_AMD64_IX86
    if (compute_type_ == SQNBIT_CompInt8) {
      data[i].QuantBDataWorkspace = packed_b_.get();
    }
#endif
    data[i].PackedQuantBData = static_cast<std::byte*>(packed_b_.get());
    data[i].QuantBScale = scales_ptr;
    data[i].QuantBZeroPoint = zero_points_data;
    data[i].Bias = bias ? bias_ptr : nullptr;
    data[i].C = c_v.data() + helper.OutputOffsets()[i];
    data[i].ldc = N;
  }
  MlasQNBitGemmBatch(M, N, K, batch_count, nbits_, block_size_, compute_type_, data.data(), workspace.get(),
                     thread_pool);
  MlasConvertFloatToHalfBuffer(c_v.data(), y_data, c_size);
  return Status::OK();
}
#endif  // end of !MLAS_F16VEC_INTRINSICS_SUPPORTED || !MLAS_TARGET_AMD64

template <>
Status MatMulNBits<float>::ComputeBUnpacked(const Tensor* a,
                                            const Tensor* b,
                                            const Tensor* scales,
                                            const Tensor* zero_points,
                                            const Tensor* reorder_idx,
                                            const Tensor* bias,
                                            Tensor* y,
                                            AllocatorPtr& allocator,
                                            concurrency::ThreadPool* thread_pool,
                                            const MatMulComputeHelper& helper) const {
  const auto* a_data = a->Data<float>();
  const uint8_t* b_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<float>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->DataRaw();
  const auto* reorder_idx_data = reorder_idx == nullptr ? nullptr : reorder_idx->Data<int32_t>();
  auto* y_data = y->MutableData<float>();

  const size_t batch_count = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(false);
  const size_t ldb = helper.Ldb(true);

  // TODO(fajin): move B dequant to prepack
  auto tmp_b_data_ptr = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(K_) * N_, true);

  if ((reorder_idx_data == nullptr) && (!zero_points || !zero_points->IsDataType<float>())) {
    // dequantize b, only 2b, 4b, and 8b quantization is supported for now
    if (this->nbits_ == 2) {
      MlasDequantizeBlockwise<float, 2>(
          tmp_b_data_ptr.get(),                           // dequantized output
          b_data,                                         // quantized input
          scales_data,                                    // quantization scales
          static_cast<const uint8_t*>(zero_points_data),  // quantization zero points
          static_cast<int32_t>(block_size_),              // quantization block size
          column_wise_quant_,                             // columnwise quantization or row-wise
          static_cast<int32_t>(K_),                       // number of rows in quantized input
          static_cast<int32_t>(N_),                       // number of columns in quantized input
          thread_pool);
    } else if (this->nbits_ == 4) {
      MlasDequantizeBlockwise<float, 4>(
          tmp_b_data_ptr.get(),                           // dequantized output
          b_data,                                         // quantized input
          scales_data,                                    // quantization scales
          static_cast<const uint8_t*>(zero_points_data),  // quantization zero points
          static_cast<int32_t>(block_size_),              // quantization block size
          column_wise_quant_,                             // columnwise quantization or row-wise
          static_cast<int32_t>(K_),                       // number of rows in quantized input
          static_cast<int32_t>(N_),                       // number of columns in quantized input
          thread_pool);
    } else {  // If it isn't 4bit, it has to be 8-bit quantization
      ORT_ENFORCE(nbits_ == 8);
      MlasDequantizeBlockwise<float, 8>(
          tmp_b_data_ptr.get(),                           // dequantized output
          b_data,                                         // quantized input
          scales_data,                                    // quantization scales
          static_cast<const uint8_t*>(zero_points_data),  // quantization zero points
          static_cast<int32_t>(block_size_),              // quantization block size
          column_wise_quant_,                             // columnwise quantization or row-wise
          static_cast<int32_t>(K_),                       // number of rows in quantized input
          static_cast<int32_t>(N_),                       // number of columns in quantized input
          thread_pool);
    }
  } else {
    // Hitting any of the below is very rare
    ORT_ENFORCE(column_wise_quant_, "Row-wise quantization is not supported for now");
    ORT_ENFORCE(nbits_ == 4,
                "Only 4b quantization is supported for unpacked compute using "
                "non-MLAS de-quantization for now");

    // !!!!!!!!!!!!!! naive implementation, need to be optimized !!!!!!!!!!!!!!
    if (zero_points && zero_points->IsDataType<float>()) {
      DequantizeBlockwise<float, float>(
          tmp_b_data_ptr.get(),                         // dequantized output
          b_data,                                       // quantized input
          scales_data,                                  // quantization scales
          static_cast<const float*>(zero_points_data),  // quantization zero points
          reorder_idx_data,
          static_cast<int32_t>(block_size_),  // quantization block size
          column_wise_quant_,                 // columnwise quantization or row-wise
          static_cast<int32_t>(K_),           // number of rows in quantized input
          static_cast<int32_t>(N_),           // number of columns in quantized input
          thread_pool);
    } else {
      DequantizeBlockwise<float, uint8_t>(
          tmp_b_data_ptr.get(),                           // dequantized output
          b_data,                                         // quantized input
          scales_data,                                    // quantization scales
          static_cast<const uint8_t*>(zero_points_data),  // quantization zero points
          reorder_idx_data,
          static_cast<int32_t>(block_size_),  // quantization block size
          column_wise_quant_,                 // columnwise quantization or row-wise
          static_cast<int32_t>(K_),           // number of rows in quantized input
          static_cast<int32_t>(N_),           // number of columns in quantized input
          thread_pool);
    }
  }
#if 0  // for debug
  auto tm_b_data_ptr_trans = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(K_) * N_);
  MlasTranspose(tmp_b_data_ptr.get(), tm_b_data_ptr_trans.get(), N_, K_);
#endif

  std::vector<MLAS_SGEMM_DATA_PARAMS> data(batch_count);
  for (size_t i = 0; i < batch_count; i++) {
    data[i].BIsPacked = false;
    data[i].A = a_data + helper.LeftOffsets()[i];
    data[i].lda = lda;
    data[i].B = tmp_b_data_ptr.get() + helper.RightOffsets()[i];
    data[i].ldb = ldb;
    data[i].C = y_data + helper.OutputOffsets()[i];
    data[i].ldc = N;
    data[i].alpha = 1.f;
    data[i].beta = 0.0f;
  }

  // if there is a bias input, copy bias values into C and set beta to 1.0f
  if (bias) {
    gsl::span<const float> bias_span = bias->DataAsSpan<float>();
    for (size_t i = 0; i < batch_count; ++i) {
      float* C_row = data[i].C;
      const size_t ldc = data[i].ldc;
      for (size_t m = 0; m < M; ++m) {
        memcpy(C_row, bias_span.data(), bias_span.size_bytes());
        C_row += ldc;
      }

      data[i].beta = 1.0f;
    }
  }

  MlasGemmBatch(CblasNoTrans, CblasTrans,
                M, N, K, data.data(), batch_count, thread_pool);

  return Status::OK();
}

template <>
Status MatMulNBits<MLFloat16>::ComputeBUnpacked(const Tensor* a,
                                                const Tensor* b,
                                                const Tensor* scales,
                                                const Tensor* zero_points,
                                                const Tensor* reorder_idx,
                                                const Tensor* bias,
                                                Tensor* y,
                                                AllocatorPtr& allocator,
                                                concurrency::ThreadPool* thread_pool,
                                                const MatMulComputeHelper& helper) const {
  const auto* a_data = a->Data<MLFloat16>();
  const uint8_t* b_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<MLFloat16>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->DataRaw();
  const auto* reorder_idx_data = reorder_idx == nullptr ? nullptr : reorder_idx->Data<int32_t>();
  auto* y_data = y->MutableData<MLFloat16>();

  const size_t batch_count = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(false);
  const size_t ldb = helper.Ldb(true);

  float* scales_ptr = nullptr;
  IAllocatorUniquePtr<float> temp_scales;
  if (!scales_fp32_) {
    auto scales_size = static_cast<size_t>(scales->Shape().Size());
    temp_scales = IAllocator::MakeUniquePtr<float>(allocator, scales_size, true);
    MlasConvertHalfToFloatBuffer(scales_data, temp_scales.get(), scales_size);
    scales_ptr = temp_scales.get();
  } else {
    scales_ptr = scales_fp32_.get();
  }

  // TODO(fajin): move B dequant to prepack
  auto tmp_b_data_ptr = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(K_) * N_, true);

  if ((reorder_idx_data == nullptr) && (!zero_points || !zero_points->IsDataType<MLFloat16>())) {
    if (nbits_ == 4) {
      MlasDequantizeBlockwise<float, 4>(
          tmp_b_data_ptr.get(),                           // dequantized output
          b_data,                                         // quantized input
          scales_ptr,                                     // quantization scales
          static_cast<const uint8_t*>(zero_points_data),  // quantization zero points
          static_cast<int32_t>(block_size_),              // quantization block size
          column_wise_quant_,                             // columnwise quantization or row-wise
          static_cast<int32_t>(K_),                       // number of rows in quantized input
          static_cast<int32_t>(N_),                       // number of columns in quantized input
          thread_pool);
    } else {  // If it isn't 4bit, it has to be 8-bit quantization
      ORT_ENFORCE(nbits_ == 8);
      MlasDequantizeBlockwise<float, 8>(
          tmp_b_data_ptr.get(),                           // dequantized output
          b_data,                                         // quantized input
          scales_ptr,                                     // quantization scales
          static_cast<const uint8_t*>(zero_points_data),  // quantization zero points
          static_cast<int32_t>(block_size_),              // quantization block size
          column_wise_quant_,                             // columnwise quantization or row-wise
          static_cast<int32_t>(K_),                       // number of rows in quantized input
          static_cast<int32_t>(N_),                       // number of columns in quantized input
          thread_pool);
    }
  } else {
    // Hitting any of the below is very rare
    ORT_ENFORCE(column_wise_quant_, "Row-wise quantization is not supported for now");
    ORT_ENFORCE(nbits_ == 4,
                "Only 4b quantization is supported for unpacked compute using "
                "non-MLAS de-quantization for now");

    // !!!!!!!!!!!!!! naive implementation, need to be optimized !!!!!!!!!!!!!!
    if (zero_points && zero_points->IsDataType<MLFloat16>()) {
      DequantizeBlockwise<float, MLFloat16>(
          tmp_b_data_ptr.get(),                             // dequantized output
          b_data,                                           // quantized input
          scales_ptr,                                       // quantization scales
          static_cast<const MLFloat16*>(zero_points_data),  // quantization zero points
          reorder_idx_data,
          static_cast<int32_t>(block_size_),  // quantization block size
          column_wise_quant_,                 // columnwise quantization or row-wise
          static_cast<int32_t>(K_),           // number of rows in quantized input
          static_cast<int32_t>(N_),           // number of columns in quantized input
          thread_pool);
    } else {
      DequantizeBlockwise<float, uint8_t>(
          tmp_b_data_ptr.get(),                           // dequantized output
          b_data,                                         // quantized input
          scales_ptr,                                     // quantization scales
          static_cast<const uint8_t*>(zero_points_data),  // quantization zero points
          reorder_idx_data,
          static_cast<int32_t>(block_size_),  // quantization block size
          column_wise_quant_,                 // columnwise quantization or row-wise
          static_cast<int32_t>(K_),           // number of rows in quantized input
          static_cast<int32_t>(N_),           // number of columns in quantized input
          thread_pool);
    }
  }
#if 0  // for debug
  auto tm_b_data_ptr_trans = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(K_) * N_);
  MlasTranspose(tmp_b_data_ptr.get(), tm_b_data_ptr_trans.get(), N_, K_);
#endif

  std::vector<MLAS_SGEMM_DATA_PARAMS> data(batch_count);

  auto a_size = static_cast<size_t>(a->Shape().Size());
  auto tmp_a_data_ptr = IAllocator::MakeUniquePtr<float>(allocator, a_size, true);
  MlasConvertHalfToFloatBuffer(a_data, tmp_a_data_ptr.get(), a_size);

  auto c_size = static_cast<size_t>(y->Shape().Size());
  auto tmp_c_ptr = IAllocator::MakeUniquePtr<float>(allocator, c_size, true);

  for (size_t i = 0; i < batch_count; i++) {
    data[i].BIsPacked = false;
    data[i].A = tmp_a_data_ptr.get() + helper.LeftOffsets()[i];
    data[i].lda = lda;
    data[i].B = tmp_b_data_ptr.get() + helper.RightOffsets()[i];
    data[i].ldb = ldb;
    data[i].C = tmp_c_ptr.get() + helper.OutputOffsets()[i];
    data[i].ldc = N;
    data[i].alpha = 1.f;
    data[i].beta = 0.0f;
  }

  // if there is a bias input, copy bias values into C and set beta to 1.0f
  if (bias) {
    float* bias_ptr = nullptr;
    const size_t bias_size = static_cast<size_t>(bias->Shape().Size());
    IAllocatorUniquePtr<float> bias_temp;
    if (!bias_fp32_) {
      bias_temp = IAllocator::MakeUniquePtr<float>(allocator, bias_size, true);
      MlasConvertHalfToFloatBuffer(bias->Data<MLFloat16>(), bias_temp.get(), bias_size);
      bias_ptr = bias_temp.get();
    } else {
      bias_ptr = bias_fp32_.get();
    }
    for (size_t i = 0; i < batch_count; ++i) {
      float* C_row = data[i].C;
      const size_t ldc = data[i].ldc;
      for (size_t m = 0; m < M; ++m) {
        std::copy(bias_ptr, bias_ptr + bias_size, C_row);
        C_row += ldc;
      }
      data[i].beta = 1.0f;
    }
  }

  MlasGemmBatch(CblasNoTrans, CblasTrans, M, N, K, data.data(), batch_count, thread_pool);
  MlasConvertFloatToHalfBuffer(tmp_c_ptr.get(), y_data, c_size);
  return Status::OK();
}

template <>
Status MatMulNBits<float>::ComputeBUnpacked3D(const Tensor* a,
                                              const Tensor* b,
                                              const Tensor* scales,
                                              const Tensor* zero_points,
                                              const Tensor* reorder_idx,
                                              const Tensor* bias,
                                              Tensor* y,
                                              AllocatorPtr& allocator,
                                              concurrency::ThreadPool* thread_pool,
                                              const MatMulComputeHelper& helper,
                                              int64_t B_nbits) const {
  ORT_ENFORCE(b != nullptr,
              "3D path (ComputeBUnpacked3D) currently requires unpacked B tensor");
  ORT_ENFORCE(scales != nullptr,
              "3D path (ComputeBUnpacked3D) requires 'scales' tensor");

  const auto* a_data = a->Data<float>();
  const uint8_t* b_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<float>();
  const void* zero_points_data_raw = zero_points ? zero_points->DataRaw() : nullptr;
  const auto* reorder_idx_data = reorder_idx ? reorder_idx->Data<int32_t>() : nullptr;
  auto* y_data = y->MutableData<float>();

  const size_t batch_count = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(false);
  const size_t ldb = helper.Ldb(true);

  // ---- צורות של B/scales/zero_points כדי לדעת את הגדלים לכל B ----
  const auto& b_q_shape = b->Shape();
  ORT_ENFORCE(b_q_shape.NumDimensions() == 4 && b_q_shape[0] == B_nbits,
              "3D path expects quantized B with shape [B,N,k_blocks,blob_size]");
  const int64_t N_q = b_q_shape[1];
  const int64_t k_blocks = b_q_shape[2];
  const int64_t blob_size = b_q_shape[3];
  ORT_ENFORCE(N_q == static_cast<int64_t>(N),
              "Mismatch in N dimension between helper and quantized B");

  const auto& s_shape = scales->Shape();
  const auto& zp_shape = zero_points ? zero_points->Shape() : TensorShape{};

  const size_t per_b_q_size =
      static_cast<size_t>(N) *
      static_cast<size_t>(k_blocks) *
      static_cast<size_t>(blob_size);

  // dequantized B נאחסן ברצף: B מטריצות כל אחת בגודל KxN
  const size_t per_b_deq_size =
      static_cast<size_t>(K_) * static_cast<size_t>(N_);
  auto tmp_b_data_ptr = IAllocator::MakeUniquePtr<float>(
      allocator,
      SafeInt<size_t>(B_nbits) * per_b_deq_size,
      true);

  // ---- 1) dequantize לכל B (fast path – בלי reorder_idx ובלי zero_points מסוג float) ----
  if ((reorder_idx_data == nullptr) && (!zero_points || !zero_points->IsDataType<float>())) {
    for (int64_t b_idx = 0; b_idx < B_nbits; ++b_idx) {
      // סלייס מהמשקלים הקוונטיים – [N,k_blocks,blob_size] עבור אותו B
      const uint8_t* b_slice =
          b_data + static_cast<size_t>(b_idx) * per_b_q_size;

      // scales עבור אותו B (אם יש ממד B, אחרת משתמשים בסקיילס משותף)
      const float* scales_slice = nullptr;
      if (s_shape.NumDimensions() == 3 && s_shape[0] == B_nbits) {
        // [B,N,k_blocks] או [B,N,1]
        const int64_t s_last = s_shape[2];  // k_blocks או 1
        const size_t per_b_s =
            static_cast<size_t>(N) * static_cast<size_t>(s_last);
        scales_slice = scales_data +
                       static_cast<size_t>(b_idx) * per_b_s;
      } else if (s_shape.NumDimensions() == 2 && s_shape[0] == B_nbits) {
        // [B,N]
        const size_t per_b_s = static_cast<size_t>(N);
        scales_slice = scales_data +
                       static_cast<size_t>(b_idx) * per_b_s;
      } else {
        // 2D legacy – אותו scales לכל B
        scales_slice = scales_data;
      }

      // zero_points עבור אותו B (אם יש ממד B, אחרת משותף)
      const uint8_t* zp_slice = nullptr;
      if (zero_points_data_raw) {
        if (zp_shape.NumDimensions() == 3 && zp_shape[0] == B_nbits) {
          // [B,N,zp_blob]
          const int64_t zp_last = zp_shape[2];
          const size_t per_b_zp =
              static_cast<size_t>(N) * static_cast<size_t>(zp_last);
          zp_slice = static_cast<const uint8_t*>(zero_points_data_raw) +
                     static_cast<size_t>(b_idx) * per_b_zp;
        } else if (zp_shape.NumDimensions() == 2 && zp_shape[0] == B_nbits) {
          // [B,N]
          const size_t per_b_zp = static_cast<size_t>(N);
          zp_slice = static_cast<const uint8_t*>(zero_points_data_raw) +
                     static_cast<size_t>(b_idx) * per_b_zp;
        } else {
          // 2D / 1D – אותו zero_point לכל B
          zp_slice = static_cast<const uint8_t*>(zero_points_data_raw);
        }
      }

      // יעד dequantized עבור אותו B – מטריצה בגודל KxN
      float* deq_slice =
          tmp_b_data_ptr.get() +
          static_cast<size_t>(b_idx) * per_b_deq_size;

      // אותו קוד כמו ב-2D, רק על הסלייסים
      if (nbits_ == 2) {
        MlasDequantizeBlockwise<float, 2>(
            deq_slice,                          // dequantized output
            b_slice,                            // quantized input
            scales_slice,                       // quantization scales
            zp_slice,                           // quantization zero points (uint8)
            static_cast<int32_t>(block_size_),  // block size
            column_wise_quant_,                 // columnwise or row-wise
            static_cast<int32_t>(K_),           // rows
            static_cast<int32_t>(N_),           // cols
            thread_pool);
      } else if (nbits_ == 4) {
        MlasDequantizeBlockwise<float, 4>(
            deq_slice,
            b_slice,
            scales_slice,
            zp_slice,
            static_cast<int32_t>(block_size_),
            column_wise_quant_,
            static_cast<int32_t>(K_),
            static_cast<int32_t>(N_),
            thread_pool);
      } else {
        ORT_ENFORCE(nbits_ == 8);
        MlasDequantizeBlockwise<float, 8>(
            deq_slice,
            b_slice,
            scales_slice,
            zp_slice,
            static_cast<int32_t>(block_size_),
            column_wise_quant_,
            static_cast<int32_t>(K_),
            static_cast<int32_t>(N_),
            thread_pool);
      }
    }
  } else {
    // כרגע בשביל פשטות – לא תומכים במצב הנדיר של reorder_idx או zero_points מסוג float ב-3D
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, NOT_IMPLEMENTED,
        "3D MatMulNBits float path currently supports only MLAS blockwise "
        "dequantization (no reorder_idx, zero_points must be uint8 or null).");
  }

#if 0  // for debug – אם תרצי לשמור, כמו בקוד המקורי
  auto tm_b_data_ptr_trans =
      IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(K_) * N_);
  MlasTranspose(tmp_b_data_ptr.get(), tm_b_data_ptr_trans.get(), N_, K_);
#endif

  // ---- 2) בונים את פרמטרי ה-GEMM בדיוק כמו 2D, רק שעכשיו RightOffsets מפנים לסלייס הנכון ----
  std::vector<MLAS_SGEMM_DATA_PARAMS> data(batch_count);
  for (size_t i = 0; i < batch_count; i++) {
    data[i].BIsPacked = false;
    data[i].A = a_data + helper.LeftOffsets()[i];
    data[i].lda = lda;
    data[i].B = tmp_b_data_ptr.get() + helper.RightOffsets()[i];
    data[i].ldb = ldb;
    data[i].C = y_data + helper.OutputOffsets()[i];
    data[i].ldc = N;
    data[i].alpha = 1.f;
    data[i].beta = 0.0f;
  }

  // ---- 3) bias – אותו קוד כמו ב-2D ----
  if (bias) {
    gsl::span<const float> bias_span = bias->DataAsSpan<float>();
    for (size_t i = 0; i < batch_count; ++i) {
      float* C_row = data[i].C;
      const size_t ldc = data[i].ldc;
      for (size_t m = 0; m < M; ++m) {
        memcpy(C_row, bias_span.data(), bias_span.size_bytes());
        C_row += ldc;
      }

      data[i].beta = 1.0f;
    }
  }

  // ---- 4) ה-GEMM עצמו – אותו דבר ----
  MlasGemmBatch(CblasNoTrans, CblasTrans,
                M, N, K, data.data(), batch_count, thread_pool);

  return Status::OK();
}

template <>
Status MatMulNBits<MLFloat16>::ComputeBUnpacked3D(
    const Tensor* a,
    const Tensor* b,
    const Tensor* scales,
    const Tensor* zero_points,
    const Tensor* reorder_idx,
    const Tensor* bias,
    Tensor* y,
    AllocatorPtr& allocator,
    concurrency::ThreadPool* thread_pool,
    const MatMulComputeHelper& helper,
    int64_t B_nbits) const {
  ORT_UNUSED_PARAMETER(a);
  ORT_UNUSED_PARAMETER(b);
  ORT_UNUSED_PARAMETER(scales);
  ORT_UNUSED_PARAMETER(zero_points);
  ORT_UNUSED_PARAMETER(reorder_idx);
  ORT_UNUSED_PARAMETER(bias);
  ORT_UNUSED_PARAMETER(y);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(thread_pool);
  ORT_UNUSED_PARAMETER(helper);
  ORT_UNUSED_PARAMETER(B_nbits);

  return ORT_MAKE_STATUS(
      ONNXRUNTIME, NOT_IMPLEMENTED,
      "3D MatMulNBits unpacked path is not implemented for MLFloat16.");
}

template <typename T1>
Status MatMulNBits<T1>::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();
  const Tensor* a = ctx->Input<Tensor>(InputIndex::A);

  // מצב prepack כפי שנקבע ב-PrePack
  bool is_b_prepacked = packed_b_size_ > 0;
  const Tensor* b = is_b_prepacked ? nullptr : ctx->Input<Tensor>(InputIndex::B);
  bool are_scales_packed = scales_are_packed_;
  const Tensor* scales = are_scales_packed ? nullptr : ctx->Input<Tensor>(InputIndex::scales);
  const Tensor* zero_points = ctx->Input<Tensor>(InputIndex::zero_points);
  const Tensor* reorder_idx = ctx->Input<Tensor>(InputIndex::g_idx);
  const Tensor* bias = ctx->Input<Tensor>(InputIndex::bias);

  // --- זיהוי B (batch) לפי צורת המשקל/סקיילים ---
  int64_t B_w = 1;
  if (!is_b_prepacked && b != nullptr) {
    const auto& b_q_shape = b->Shape();
    if (b_q_shape.NumDimensions() == 4) {
      B_w = b_q_shape[0];  // [B, N, k_blocks, blob_size]
    }
  }

  int64_t B_s = 1;
  if (!are_scales_packed && scales != nullptr) {
    const auto& s_shape = scales->Shape();
    const int s_rank = s_shape.NumDimensions();
    if (s_rank == 3) {
      B_s = s_shape[0];  // [B, N, k_blocks]
    } else if (s_rank == 2 && s_shape[0] != static_cast<int64_t>(N_)) {
      B_s = s_shape[0];  // [B, N] (ללא קיבוץ)
    }
  }

  const int64_t B_nbits = std::max(B_w, B_s);

  // --- אם זיהינו 3D: נטרל prepack ואז קַשר מחדש את המצביעים --- ★
  if (B_nbits > 1) {
    if (packed_b_size_ > 0 || scales_are_packed_) {
      LOGS(ctx->Logger(), WARNING)
          << "MatMulNBits: disabling prepacked state for batched weights/scales.";
    }

    // לאחר נטרול, לקרוא מחדש את הקלטים כדי לקבל את הטנזורים עצמם ★
    is_b_prepacked = false;
    are_scales_packed = false;
    b = ctx->Input<Tensor>(InputIndex::B);
    scales = ctx->Input<Tensor>(InputIndex::scales);
  }

  // --- הבדיקה/ולידציה חייבת להיות אחרי הנטרול וה-rebind --- ★
  ORT_RETURN_IF_ERROR(matmul_nbits_helper::CheckInputs<Tensor>(
      a, b, scales, zero_points, reorder_idx, bias, N_, K_, block_size_, nbits_));

  // *** אל תחזירי יותר ORT_ENFORCE על prepack ב-3D ***

  // Prepare logical shape of B for MatMulComputeHelper
  TensorShape b_shape = (B_nbits == 1)
                            ? TensorShape({static_cast<int64_t>(N_), static_cast<int64_t>(K_)})
                            : TensorShape({B_nbits, static_cast<int64_t>(N_), static_cast<int64_t>(K_)});

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, /*transB=*/true));

  Tensor* y = ctx->Output(0, helper.OutputShape());
  if (y->Shape().Size() == 0) {
    return Status::OK();
  }

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));

  const bool has_single_b_matrix =
      std::all_of(helper.RightOffsets().begin(), helper.RightOffsets().end(),
                  [](size_t offset) { return offset == 0; });

  if (B_nbits == 1 && has_single_b_matrix && packed_b_ &&
      MlasIsQNBitGemmAvailable(nbits_, block_size_, compute_type_)) {
    return ComputeBPacked(a, scales, zero_points, bias, y, allocator, thread_pool, helper);
  }

  LOGS(ctx->Logger(), INFO)
      << "Falling back to unpacked compute mode for MatMulNBits.";

  if (B_nbits > 1) {
    // נתיב 3D (unpacked)
    return ComputeBUnpacked3D(a, b, scales, zero_points, reorder_idx, bias,
                              y, allocator, thread_pool, helper, B_nbits);
  }

  // נתיב 2D (unpacked)
  return ComputeBUnpacked(a, b, scales, zero_points, reorder_idx, bias,
                          y, allocator, thread_pool, helper);
}

#define REGISTER_MatMulNBits(T1)                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      MatMulNBits,                                                       \
      kMSDomain,                                                         \
      1,                                                                 \
      T1,                                                                \
      kCpuExecutionProvider,                                             \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())       \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())  \
          .TypeConstraint("T3", {DataTypeImpl::GetTensorType<uint8_t>(), \
                                 DataTypeImpl::GetTensorType<T1>()})     \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()), \
      MatMulNBits<T1>);

REGISTER_MatMulNBits(float);
REGISTER_MatMulNBits(MLFloat16);

}  // namespace contrib
}  // namespace onnxruntime
