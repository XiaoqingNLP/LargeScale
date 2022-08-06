import torch
from cpm_kernels.kernels import gemm_int8
from bminf.kernels import gemm_calc_scale, gemm_round, gemm_scale, gemm_scale_round


class INT8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor, quant_w: torch.Tensor, scale_w: torch.Tensor):
        """
        Args:
            inp:    (..., in_features)          dtype
            weight: (out_features, in_features) dtype
            scale:  (out_features,)             dtype
        Returns:
            out:    (..., out_features)
        """
        ctx.inp_shape = inp.size()
        ctx.weight_shape = weight.size()
        x = inp.contiguous().view(-1, weight.size(1))
        scale_x = gemm_calc_scale(x)  # (...,)
        quant_x = gemm_round(x, scale_x)  # (..., in_features)    int8

        M = quant_x.size(0)
        K = quant_x.size(1)
        N = quant_w.size(0)

        with torch.cuda.device(inp.device):
            out = torch.empty(M, N, dtype=torch.int32, device="cuda")  # (..., out_features)
            gemm_int8(
                N, K, M,
                1, 1,
                True, False,
                quant_w.data_ptr(), quant_x.data_ptr(),
                out.data_ptr(),
                torch.cuda.current_stream().cuda_stream
            )
        out = gemm_scale(out, scale_x, scale_w, inp.dtype)

        ctx.save_for_backward(quant_x, scale_x, quant_w, scale_w)
        return out.view(*(ctx.inp_shape[:-1] + (N,)))

    @staticmethod
    def backward(ctx, grad_f: torch.Tensor):
        quant_x, scale_x, quant_w, scale_w = ctx.saved_tensors
        grad_f = grad_f.contiguous().view(-1, quant_w.size(0))  # (..., out_features)

        quant_grad_f, scale_grad_f = gemm_scale_round(grad_f, scale_w)

        grad_input = torch.empty(grad_f.size(0), quant_w.size(1), dtype=torch.int32, device="cuda")
        gemm_int8(
            quant_w.size(1), quant_w.size(0), quant_grad_f.size(0),
            1, 1,
            True, False,
            # Here we need a handwritten transpose to enable TensorCore
            quant_w.transpose(0, 1).contiguous().data_ptr(), quant_grad_f.data_ptr(),
            grad_input.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )
        grad_input = gemm_scale(grad_input, scale_grad_f, None, grad_f.dtype)

        quant_grad_f, scale_grad_f = gemm_scale_round(grad_f.transpose(0, 1).contiguous(), scale_x)

        grad_weight = torch.empty(grad_f.size(1), quant_w.size(1), dtype=torch.int32, device="cuda")
        gemm_int8(
            quant_x.size(1), quant_x.size(0), quant_grad_f.size(0),
            1, 1,
            True, False,
            # Here we need a handwritten transpose to enable TensorCore
            quant_x.transpose(0, 1).contiguous().data_ptr(), quant_grad_f.data_ptr(),
            grad_weight.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )
        grad_weight = gemm_scale(grad_weight, scale_grad_f, None, grad_f.dtype)

        return grad_input.view(ctx.inp_shape), grad_weight.view(ctx.weight_shape), None, None
