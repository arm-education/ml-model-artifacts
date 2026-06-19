# Model Explorer artifacts

This repository contains model artifacts for the Arm Learning Path
**Visualize and understand ExecuTorch, TOSA, and Neural Graphics Models with
Google's Model Explorer**.

The artifacts are intended to be opened in Google Model Explorer with the Arm
Model Explorer adapters:

- `pte-adapter-model-explorer` for ExecuTorch `.pte` programs
- `tosa-adapter-model-explorer` for TOSA `.tosa` intermediate representations
- `vgf-adapter-model-explorer` for Vulkan Graph Format `.vgf` artifacts
- ETDump `.etdp` and ETRecord `.etrecord` files for debugging ExecuTorch
  execution traces

## Git LFS

This repository uses Git LFS for model artifacts. Install Git LFS before
cloning, or run `git lfs pull` after cloning to download the actual `.pte`,
`.tosa`, `.vgf`, `.etdp`, and `.etrecord` files.

## Repository layout

```text
model-explorer-artifacts/
├── LICENSE.md
├── README.md
├── etdump/
│   ├── mobilenetv2_fp32_ethosu.etdp
│   ├── mobilenetv2_int8_ethosu.etdp
│   ├── mobilenetv2_lrn_int8_ethosu.etdp
│   ├── opt125m_portable.etdp
│   └── opt125m_xnnpack.etdp
├── etrecord/
│   ├── mobilenetv2_fp32_ethosu.etrecord
│   ├── mobilenetv2_int8_ethosu.etrecord
│   ├── mobilenetv2_lrn_int8_ethosu.etrecord
│   ├── opt125m_portable.etrecord
│   └── opt125m_xnnpack.etrecord
├── pte/
│   ├── add_sigmoid_vgf.pte
│   ├── mv2_cortex_m.pte
│   ├── mv2_fp32_ethos_u85.pte
│   ├── mv2_int8_ethos_u85.pte
│   ├── mv2_lrn_int8_ethos_u85.pte
│   ├── opt125m_cortex_a_portable.pte
│   ├── opt125m_cortex_a_xnnpack.pte
│   ├── small_upscaler_ptq_vgf.pte
│   └── small_upscaler_qat_vgf.pte
├── tosa/
│   ├── mv2_fp32.tosa
│   ├── mv2_int8.tosa
│   ├── mv2_lrn_int8_1.tosa
│   ├── mv2_lrn_int8_2.tosa
│   ├── small_upscaler_ptq.tosa
│   └── small_upscaler_qat.tosa
└── vgf/
    ├── add_sigmoid.vgf
    ├── small_upscaler_ptq.vgf
    └── small_upscaler_qat.vgf
```

## Artifact groups

### ETDump artifacts

The `etdump/` directory contains ExecuTorch debug data files. Use these with
ExecuTorch debugging tools to inspect runtime events and relate execution
behavior back to exported program artifacts.

| File | Purpose |
| --- | --- |
| `mobilenetv2_fp32_ethosu.etdp` | Debug data for the MobileNetV2 floating-point Ethos-U85 run. |
| `mobilenetv2_int8_ethosu.etdp` | Debug data for the MobileNetV2 int8 Ethos-U85 run. |
| `mobilenetv2_lrn_int8_ethosu.etdp` | Debug data for the fragmented MobileNetV2 int8 Ethos-U85 run. |
| `opt125m_portable.etdp` | Debug data for the OPT-125M portable-kernel run. |
| `opt125m_xnnpack.etdp` | Debug data for the OPT-125M XNNPACK-delegated run. |

### ETRecord artifacts

The `etrecord/` directory contains ExecuTorch record files. Use these with
ExecuTorch debugging tools to map runtime trace data to exported programs,
delegate regions, and operator-level execution details.

| File | Purpose |
| --- | --- |
| `mobilenetv2_fp32_ethosu.etrecord` | Record file for the MobileNetV2 floating-point Ethos-U85 export. |
| `mobilenetv2_int8_ethosu.etrecord` | Record file for the MobileNetV2 int8 Ethos-U85 export. |
| `mobilenetv2_lrn_int8_ethosu.etrecord` | Record file for the fragmented MobileNetV2 int8 Ethos-U85 export. |
| `opt125m_portable.etrecord` | Record file for the OPT-125M portable-kernel export. |
| `opt125m_xnnpack.etrecord` | Record file for the OPT-125M XNNPACK-delegated export. |

### PTE artifacts

The `pte/` directory contains ExecuTorch program files. Use these with the PTE
adapter to inspect deployment graphs, delegate regions, backend partitioning,
and CPU fallback.

| File | Purpose |
| --- | --- |
| `add_sigmoid_vgf.pte` | Small ExecuTorch program using the Arm VGF backend path. |
| `mv2_cortex_m.pte` | MobileNetV2 Cortex-M ExecuTorch artifact. |
| `mv2_fp32_ethos_u85.pte` | MobileNetV2 floating-point Ethos-U85 artifact. |
| `mv2_int8_ethos_u85.pte` | MobileNetV2 int8 Ethos-U85 artifact. |
| `mv2_lrn_int8_ethos_u85.pte` | MobileNetV2 int8 Ethos-U85 artifact with fragmented lowering. |
| `opt125m_cortex_a_portable.pte` | OPT-125M Cortex-A artifact using portable kernels. |
| `opt125m_cortex_a_xnnpack.pte` | OPT-125M Cortex-A artifact with XNNPACK delegation. |
| `small_upscaler_ptq_vgf.pte` | Small upscaler post-training quantized artifact using the Arm VGF backend path. |
| `small_upscaler_qat_vgf.pte` | Small upscaler quantization-aware trained artifact using the Arm VGF backend path. |

### TOSA artifacts

The `tosa/` directory contains TOSA intermediate representations. Use these with
the TOSA adapter to inspect lowered operators, tensor shapes, quantized types,
graph splits, and optimization opportunities.

| File | Purpose |
| --- | --- |
| `mv2_fp32.tosa` | MobileNetV2 floating-point TOSA graph. |
| `mv2_int8.tosa` | MobileNetV2 int8 TOSA graph. |
| `mv2_lrn_int8_1.tosa` | First TOSA graph partition for the fragmented MobileNetV2 int8 lowering example. |
| `mv2_lrn_int8_2.tosa` | Second TOSA graph partition for the fragmented MobileNetV2 int8 lowering example. |
| `small_upscaler_ptq.tosa` | Small upscaler TOSA graph produced from post-training quantization. |
| `small_upscaler_qat.tosa` | Small upscaler TOSA graph produced from quantization-aware training. |

### VGF artifacts

The `vgf/` directory contains Vulkan Graph Format artifacts. Use these with the
VGF adapter to inspect graph connectivity, tensor metadata, constants, and
SPIR-V graph modules used by Vulkan ML workflows.

| File | Purpose |
| --- | --- |
| `add_sigmoid.vgf` | Small add/sigmoid VGF graph. |
| `small_upscaler_ptq.vgf` | Small upscaler VGF artifact produced from post-training quantization. |
| `small_upscaler_qat.vgf` | Small upscaler VGF artifact produced from quantization-aware training. |

## Use with Model Explorer

Install Model Explorer and the Arm adapters in a Python virtual environment:

```bash
python -m venv model_explorer_env
source model_explorer_env/bin/activate
pip install ai-edge-model-explorer
pip install pte-adapter-model-explorer
pip install tosa-adapter-model-explorer
pip install vgf-adapter-model-explorer
```

Launch Model Explorer with all three adapters:

```bash
model-explorer --extensions=pte_adapter_model_explorer,tosa_adapter_model_explorer,vgf_adapter_model_explorer
```

Then open an artifact from the `pte/`, `tosa/`, or `vgf/` directory. Use the
matching files from `etdump/` and `etrecord/` with ExecuTorch debugging tools
when you need runtime trace context.

Some artifacts are large, especially the OPT-125M `.pte` and `.etrecord` files.
They may take longer to download, load, and render than the smaller examples.

## License

This repository uses the Arm Education End User License Agreement for teaching
and learning content. See [LICENSE.md](LICENSE.md).
