# -*- coding: utf-8 -*-
"""Stable Diffusion v1.5 文生图 WebUI 演示（不加载真实模型权重，仅前端展示）。"""
from __future__ import annotations

import gradio as gr
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent / "images"
PLACEHOLDER_IMAGE = IMAGES_DIR / "stable_diffusion_v1_5_model_page.png"

def fake_load_model():
    return "模型状态：Stable Diffusion v1.5 已就绪（演示模式，未加载真实权重）"

def fake_generate(prompt: str, steps: int, guidance_scale: float):
    if not (prompt or "").strip():
        return None, "请输入文本提示词后再点击生成。"
    if PLACEHOLDER_IMAGE.exists():
        return str(PLACEHOLDER_IMAGE), f"【演示】已根据提示「{prompt[:50]}…」模拟生成（未加载真实模型）。步数={steps}，引导系数={guidance_scale}。"
    return None, f"【演示】已接收提示「{prompt[:80]}」，步数={steps}，引导系数={guidance_scale}。加载真实模型后此处将显示生成图像。"

def build_ui():
    with gr.Blocks(title="Stable Diffusion v1.5 文生图 WebUI") as demo:
        gr.Markdown("## Stable Diffusion v1.5 文生图 · WebUI 演示")
        gr.Markdown("本界面以交互方式展示 Stable Diffusion v1.5 文生图（Text-to-Image）的典型使用流程，包括模型加载状态与根据文本提示生成图像的展示。")
        with gr.Row():
            load_btn = gr.Button("加载模型（演示）", variant="primary")
            status_box = gr.Textbox(label="模型状态", value="尚未加载", interactive=False)
        load_btn.click(fn=fake_load_model, outputs=status_box)
        with gr.Tabs():
            with gr.Tab("文生图"):
                gr.Markdown("在下方输入文本提示词，设置采样步数与引导系数，点击生成即可查看演示结果。")
                prompt_in = gr.Textbox(label="文本提示词", placeholder="例如：a photo of an astronaut riding a horse on mars", lines=3)
                with gr.Row():
                    steps_slider = gr.Slider(minimum=10, maximum=100, value=50, step=1, label="采样步数")
                    guidance_slider = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.5, label="引导系数")
                gen_btn = gr.Button("生成图像（演示）", variant="primary")
                img_out = gr.Image(label="生成结果", type="filepath")
                msg_out = gr.Textbox(label="说明", lines=2, interactive=False)
                gen_btn.click(fn=fake_generate, inputs=[prompt_in, steps_slider, guidance_slider], outputs=[img_out, msg_out])
        gr.Markdown("---\n*说明：当前为轻量级演示界面，未实际下载与加载 Stable Diffusion v1.5 模型参数。*")
    return demo

def main():
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=17960, share=False)

if __name__ == "__main__":
    main()
