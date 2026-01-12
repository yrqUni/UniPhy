import os
import re

TARGET_FILE = 'GridSample.py'

def main():
    if not os.path.exists(TARGET_FILE):
        print(f"❌ 错误: 找不到文件 {TARGET_FILE}")
        return

    print(f"🔄 正在读取 {TARGET_FILE}...")
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # =========================================================
    # 修复 1: Forward 函数中的 Stride 映射
    # =========================================================
    # 查找旧的 stride 获取逻辑
    old_forward_stride = """        use_res_flow = res_flows is not None
        r_stride = res_flows.stride() if use_res_flow else (0,0,0,0,0,0)

        fused_pscan_forward_kernel_2d[grid_dim](
            images, cum_flows, res_flows, out, mask, decay_dist,
            B, C, L, H, W, T_chunk, K_chunk,
            images.stride(0), images.stride(1), images.stride(2), images.stride(3), images.stride(4),
            cum_flows.stride(0), cum_flows.stride(1), cum_flows.stride(2), cum_flows.stride(3), cum_flows.stride(4),
            r_stride[0], r_stride[1], r_stride[2], r_stride[3], r_stride[4], r_stride[5],"""

    # 新的逻辑：显式解包 5D stride 并插入 0 作为 T 维度的 stride
    new_forward_stride = """        use_res_flow = res_flows is not None
        if use_res_flow:
            s = res_flows.stride()
            # res_flows is (B, K, C, H, W), map to (B, T, K, C, H, W) with stride_t=0
            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w = s[0], 0, s[1], s[2], s[3], s[4]
        else:
            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w = 0, 0, 0, 0, 0, 0

        fused_pscan_forward_kernel_2d[grid_dim](
            images, cum_flows, res_flows, out, mask, decay_dist,
            B, C, L, H, W, T_chunk, K_chunk,
            images.stride(0), images.stride(1), images.stride(2), images.stride(3), images.stride(4),
            cum_flows.stride(0), cum_flows.stride(1), cum_flows.stride(2), cum_flows.stride(3), cum_flows.stride(4),
            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w,"""

    # 执行替换 (使用 replace，注意 Python 缩进非常敏感，这里假设缩进是匹配的)
    # 如果直接 replace 失败，尝试正则或手动定位
    if old_forward_stride in content:
        content = content.replace(old_forward_stride, new_forward_stride)
        print("✅ Forward Stride 逻辑已修复")
    else:
        # 尝试去掉空白字符进行匹配的备用方案（防止空格数不一致）
        # 这里为了稳妥，我们使用较短的特征串进行替换
        pass 
        # (如果上述完全匹配失败，可能因为你之前手动修改过格式，我们尝试分段替换)
        
    # =========================================================
    # 修复 2: Backward 函数中的 Stride 映射
    # =========================================================
    old_backward_stride = """        grad_output = grad_output.contiguous()
        use_res_flow = res_flows is not None
        r_stride = res_flows.stride() if use_res_flow else (0,0,0,0,0,0)

        fused_pscan_backward_kernel_2d[grid_dim](
            grad_output, images, cum_flows, res_flows, mask, decay_dist,
            grad_images, grad_cum_flows, grad_res_flows,
            B, C, L, H, W, T_chunk, K_chunk,
            images.stride(0), images.stride(1), images.stride(2), images.stride(3), images.stride(4),
            cum_flows.stride(0), cum_flows.stride(1), cum_flows.stride(2), cum_flows.stride(3), cum_flows.stride(4),
            r_stride[0], r_stride[1], r_stride[2], r_stride[3], r_stride[4], r_stride[5],"""

    new_backward_stride = """        grad_output = grad_output.contiguous()
        use_res_flow = res_flows is not None
        if use_res_flow:
            s = res_flows.stride()
            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w = s[0], 0, s[1], s[2], s[3], s[4]
        else:
            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w = 0, 0, 0, 0, 0, 0

        fused_pscan_backward_kernel_2d[grid_dim](
            grad_output, images, cum_flows, res_flows, mask, decay_dist,
            grad_images, grad_cum_flows, grad_res_flows,
            B, C, L, H, W, T_chunk, K_chunk,
            images.stride(0), images.stride(1), images.stride(2), images.stride(3), images.stride(4),
            cum_flows.stride(0), cum_flows.stride(1), cum_flows.stride(2), cum_flows.stride(3), cum_flows.stride(4),
            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w,"""

    if old_backward_stride in content:
        content = content.replace(old_backward_stride, new_backward_stride)
        print("✅ Backward Stride 逻辑已修复")
    else:
        # 如果长段匹配失败，尝试更宽松的正则替换
        # Forward Pattern
        pattern_fwd = re.compile(
            r"r_stride\s*=\s*res_flows\.stride\(\)\s*if\s*use_res_flow\s*else\s*\(0,0,0,0,0,0\)\s*"
            r"fused_pscan_forward_kernel_2d\[grid_dim\]\(\s*"
            r"(.*?)"
            r"r_stride\[0\],\s*r_stride\[1\],\s*r_stride\[2\],\s*r_stride\[3\],\s*r_stride\[4\],\s*r_stride\[5\],",
            re.DOTALL
        )
        
        replacement_fwd = (
            "if use_res_flow:\n"
            "            s = res_flows.stride()\n"
            "            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w = s[0], 0, s[1], s[2], s[3], s[4]\n"
            "        else:\n"
            "            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w = 0, 0, 0, 0, 0, 0\n\n"
            "        fused_pscan_forward_kernel_2d[grid_dim](\n"
            r"            \1"
            "rs_b, rs_t, rs_k, rs_c, rs_h, rs_w,"
        )
        
        if pattern_fwd.search(content):
            content = pattern_fwd.sub(replacement_fwd, content)
            print("✅ Forward Stride 逻辑已修复 (Regex)")

        # Backward Pattern
        pattern_bwd = re.compile(
            r"r_stride\s*=\s*res_flows\.stride\(\)\s*if\s*use_res_flow\s*else\s*\(0,0,0,0,0,0\)\s*"
            r"fused_pscan_backward_kernel_2d\[grid_dim\]\(\s*"
            r"(.*?)"
            r"r_stride\[0\],\s*r_stride\[1\],\s*r_stride\[2\],\s*r_stride\[3\],\s*r_stride\[4\],\s*r_stride\[5\],",
            re.DOTALL
        )
        
        replacement_bwd = (
            "if use_res_flow:\n"
            "            s = res_flows.stride()\n"
            "            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w = s[0], 0, s[1], s[2], s[3], s[4]\n"
            "        else:\n"
            "            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w = 0, 0, 0, 0, 0, 0\n\n"
            "        fused_pscan_backward_kernel_2d[grid_dim](\n"
            r"            \1"
            "rs_b, rs_t, rs_k, rs_c, rs_h, rs_w,"
        )
        
        if pattern_bwd.search(content):
            content = pattern_bwd.sub(replacement_bwd, content)
            print("✅ Backward Stride 逻辑已修复 (Regex)")


    # 写入文件
    if content != original_content:
        backup_file = TARGET_FILE + ".bak_stride"
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"💾 已备份原文件至 {backup_file}")
        
        with open(TARGET_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"🎉 成功！{TARGET_FILE} stride 越界问题已修复。")
    else:
        print("⚠️ 未进行任何更改。可能是代码格式与脚本预期不符，或者已经修复。")

if __name__ == "__main__":
    main()

