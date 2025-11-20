// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"
#include "core/util/shape_checker.h"

namespace onnxruntime {
namespace contrib {
namespace matmul_nbits_helper {

// template <typename T = Tensor>
// Status CheckInputs(const T* /*activation*/,
//                    const T* quantized_weight,
//                    const T* scales,
//                    const T* zero_points,
//                    const T* group_index,
//                    const T* bias,
//                    int64_t n,
//                    int64_t k,
//                    int64_t block_size,
//                    int64_t bits) {
//   // activation (A)
//   // quantized_weight (B) : (N, k_blocks, blob_size), or null after prepacking.
//   //                        k_blocks = (K + block_size - 1) / block_size
//   //                        blob_size = block_size * bits / 8
//   // scales               : (N, k_blocks)
//   // zero_points          : (N, (k_blocks * bits + 7) / 8) for uint8, (N, k_blocks) for other types, or null
//   // group_index          : (K) or (k_blocks * block_size), or null
//   // bias                 : (N), or null
//   // Note that scales and zero_points can be 1D for backward compatibility.
//   if (bits != 2 && bits != 4 && bits != 8) {
//     return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "bits should be 2, 4 or 8, got ", bits);
//   }

//   if (block_size < 16 || (block_size & (block_size - 1)) != 0) {
//     return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
//                            "block_size must be a power of 2, and >= 16. Got ", block_size);
//   }

//   int64_t k_blocks = (k + block_size - 1) / block_size;
//   int64_t blob_size = block_size * bits / 8;

//   int64_t B = 1;
// if (quantized_weight != nullptr) {
//   const auto& wshape = quantized_weight->Shape();
//   if (wshape.NumDimensions() == 4) {
//     B = wshape[0]; // תלת־ממד: Wq[B, N, k_blocks, blob_size]
//   }
// }

//   ASSERT_TENSOR_SHAPE(quantized_weight, make_shape(n, k_blocks, blob_size));

//   // 1D shape is for backward compatibility for existing models.
//   ASSERT_TENSOR_SHAPE_2(scales, make_shape(n * k_blocks), make_shape(n, k_blocks));

//   if (zero_points != nullptr) {
//     if (zero_points->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
//       const int64_t zero_point_blob_size = (k_blocks * bits + 7) / 8;

//       ASSERT_TENSOR_SHAPE_2(zero_points, make_shape(n * zero_point_blob_size), make_shape(n, zero_point_blob_size));
//     } else {
//       if (zero_points->GetElementType() != scales->GetElementType()) {
//         return ORT_MAKE_STATUS(
//             ONNXRUNTIME, INVALID_ARGUMENT,
//             "Input 'zero_points' and 'scales' should have the same data type when zero_points is not uint8");
//       }

//       ASSERT_TENSOR_SHAPE_2(zero_points, make_shape(n * k_blocks), make_shape(n, k_blocks));
//     }
//   }

//   // Group_index shall be 1D of K, or K padded to multiple of block_size
//   ASSERT_TENSOR_SHAPE_2(group_index, make_shape(k), make_shape(k_blocks * block_size));

//   ASSERT_TENSOR_SHAPE(bias, make_shape(n));

//   return Status::OK();
// }

template <typename T = Tensor>
Status CheckInputs(const T* /*activation*/,
                   const T* quantized_weight,
                   const T* scales,
                   const T* zero_points,
                   const T* group_index,
                   const T* bias,
                   int64_t n,
                   int64_t k,
                   int64_t block_size,
                   int64_t bits) {
  if (bits != 2 && bits != 4 && bits != 8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "bits should be 2, 4 or 8, got ", bits);
  }
  if (block_size < 16 || (block_size & (block_size - 1)) != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "block_size must be a power of 2, and >= 16. Got ", block_size);
  }

  const int64_t k_blocks = (k + block_size - 1) / block_size;
  const int64_t blob_size = block_size * bits / 8;

  int64_t B = 1;
  if (quantized_weight != nullptr) {
    const auto& ws = quantized_weight->Shape();
    if (ws.NumDimensions() == 4) {
      B = ws[0];
    }
  }
  if (B == 1 && scales != nullptr) {
    const auto& ss = scales->Shape();
    if (ss.NumDimensions() == 3) {
      B = ss[0];
    }
  }

  if (quantized_weight != nullptr) {
    const auto& ws = quantized_weight->Shape();
    const int wr = ws.NumDimensions();

    const bool ok_wq_2d =
        (wr == 3 && ws[0] == n && ws[1] == k_blocks && ws[2] == blob_size);
    const bool ok_wq_3d =
        (wr == 4 && ws[0] == B && ws[1] == n && ws[2] == k_blocks && ws[3] == blob_size);

    if (!(ok_wq_2d || ok_wq_3d)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "quantized_weight must be (N,k_blocks,blob_size) or (B,N,k_blocks,blob_size). Got ",
                             ws.ToString());
    }
  }

  // === 2) scales ===
  {
    const auto& ss = scales->Shape();
    const int sr = ss.NumDimensions();

    // תמיכה לאחור (הבדיקה שהייתה ב-ASSERT_TENSOR_SHAPE_2):
    const bool ok_scales_legacy =
        (sr == 1 && ss[0] == n * k_blocks) ||          // [N*k_blocks]
        (sr == 2 && ss[0] == n && ss[1] == k_blocks);  // [N,k_blocks]

    // הוספת תמיכה ל-3D:
    const bool ok_scales_3d =
        (sr == 3 && ss[0] == B && ss[1] == n &&
         (ss[2] == k_blocks || ss[2] == 1)) ||  // [B,N,k_blocks] או [B,N,1]
        (sr == 2 && ss[0] == B && ss[1] == n);  // [B,N] ללא group-wise

    if (!(ok_scales_legacy || ok_scales_3d)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "scales bad shape. Accept {N*k_blocks}/{N,k_blocks} (2D) or {B,N,k_blocks}/{B,N,1}/{B,N} (3D). Got ",
                             ss.ToString());
    }
  }

  // === 3) zero_points (אם יש) ===
  if (zero_points != nullptr) {
    const auto& zs = zero_points->Shape();
    const int zr = zs.NumDimensions();

    if (zero_points->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
      // UINT8 = אריזה ביטית; נדרש zp_blob
      const int64_t zp_blob = (k_blocks * bits + 7) / 8;

      const bool ok_zp_u8 =
          (zr == 1 && zs[0] == n * zp_blob) ||                        // [N*zp_blob] (תמיכה לאחור)
          (zr == 2 && zs[0] == n && zs[1] == zp_blob) ||              // [N,zp_blob]
          (zr == 3 && zs[0] == B && zs[1] == n && zs[2] == zp_blob);  // [B,N,zp_blob]

      if (!ok_zp_u8) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "zero_points(uint8) bad shape. Accept {N*zp_blob}/{N,zp_blob} or {B,N,zp_blob}. Got ",
                               zs.ToString());
      }
    } else {
      // non-uint8: אותו dtype כמו scales + אותן צורות מותרות כמו scales
      if (zero_points->GetElementType() != scales->GetElementType()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'zero_points' and 'scales' should have the same data type when zero_points is not uint8");
      }

      const bool ok_zp_legacy =
          (zr == 1 && zs[0] == n * k_blocks) ||          // [N*k_blocks]
          (zr == 2 && zs[0] == n && zs[1] == k_blocks);  // [N,k_blocks]

      const bool ok_zp_3d =
          (zr == 3 && zs[0] == B && zs[1] == n &&
           (zs[2] == k_blocks || zs[2] == 1)) ||  // [B,N,k_blocks] או [B,N,1]
          (zr == 2 && zs[0] == B && zs[1] == n);  // [B,N]

      if (!(ok_zp_legacy || ok_zp_3d)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "zero_points bad shape. Accept {N*k_blocks}/{N,k_blocks} or {B,N,k_blocks}/{B,N,1}/{B,N}. Got ",
                               zs.ToString());
      }
    }
  }

  // === group_index / bias (ללא שינוי) ===
  ASSERT_TENSOR_SHAPE_2(group_index, make_shape(k), make_shape(k_blocks * block_size));
  ASSERT_TENSOR_SHAPE(bias, make_shape(n));

  return Status::OK();
}

}  // namespace matmul_nbits_helper
}  // namespace contrib
}  // namespace onnxruntime
