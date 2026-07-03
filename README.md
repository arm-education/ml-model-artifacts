# Arm ML model artifacts

This repository contains model artifacts used by Arm Learning Paths that teach
model analysis, graph inspection, deployment-artifact inspection, and
target-aware ML workflow concepts.

These artifacts are companion material for learning-path exercises and
explanations. Different Learning Paths may use different subsets of the
repository depending on whether the focus is MLIA analysis, Model Explorer
inspection, ExecuTorch runtime traces, TOSA lowering, TensorFlow Lite models, or
Vulkan ML artifacts.

## Git LFS

This repository uses Git LFS for model artifacts. Install Git LFS before
cloning, or run `git lfs pull` after cloning to download the actual `.pt2`,
`.pte`, `.tflite`, `.tosa`, `.vgf`, `.etdp`, and `.etrecord` files.

## Repository layout

```text
ml-model-artifacts/
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
├── pt2/
│   └── mv2_fp32.pt2
├── tflite/
│   ├── mv2_fp32.tflite
│   ├── mv2_int8.tflite
│   └── mv2_lrn_int8.tflite
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

The `etdump/` directory contains ExecuTorch debug data files. Open these with
the ExecuTorch Model Explorer extension to inspect runtime events and relate
execution behavior back to exported program artifacts.

| File | Purpose |
| --- | --- |
| `mobilenetv2_fp32_ethosu.etdp` | Debug data for the MobileNetV2 floating-point Ethos-U85 run. |
| `mobilenetv2_int8_ethosu.etdp` | Debug data for the MobileNetV2 int8 Ethos-U85 run. |
| `mobilenetv2_lrn_int8_ethosu.etdp` | Debug data for the fragmented MobileNetV2 int8 Ethos-U85 run. |
| `opt125m_portable.etdp` | Debug data for the OPT-125M portable-kernel run. |
| `opt125m_xnnpack.etdp` | Debug data for the OPT-125M XNNPACK-delegated run. |

### ETRecord artifacts

The `etrecord/` directory contains ExecuTorch record files. Open these with the
ExecuTorch Model Explorer extension to map runtime trace data to exported
programs, delegate regions, and operator-level execution details.

| File | Purpose |
| --- | --- |
| `mobilenetv2_fp32_ethosu.etrecord` | Record file for the MobileNetV2 floating-point Ethos-U85 export. |
| `mobilenetv2_int8_ethosu.etrecord` | Record file for the MobileNetV2 int8 Ethos-U85 export. |
| `mobilenetv2_lrn_int8_ethosu.etrecord` | Record file for the fragmented MobileNetV2 int8 Ethos-U85 export. |
| `opt125m_portable.etrecord` | Record file for the OPT-125M portable-kernel export. |
| `opt125m_xnnpack.etrecord` | Record file for the OPT-125M XNNPACK-delegated export. |

### PTE artifacts

The `pte/` directory contains ExecuTorch program files. Use these with the PTE
adapter (part of the Model Explorer ExecuTorch extension) to inspect deployment graphs, delegate regions, backend partitioning,
and CPU fallback. You can also use these with MLIA.

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

### PT2 artifacts

The `pt2/` directory contains PyTorch exported programs for use with MLIA
converter plugins or PyTorch/ExecuTorch-oriented workflows.

| File | Purpose |
| --- | --- |
| `mv2_fp32.pt2` | MobileNetV2 floating-point PyTorch exported program. |

### TFLite artifacts

The `tflite/` directory contains TensorFlow Lite MobileNetV2 artifacts for use with Model Explorer or MLIA

| File | Purpose |
| --- | --- |
| `mv2_fp32.tflite` | MobileNetV2 floating-point TensorFlow Lite model. |
| `mv2_int8.tflite` | MobileNetV2 full-integer TensorFlow Lite model with int8 input and output tensors. |
| `mv2_lrn_int8.tflite` | MobileNetV2 full-integer TensorFlow Lite model with an inserted local response normalization operation. |

### TOSA artifacts

The `tosa/` directory contains TOSA intermediate representations. Use these with
the TOSA adapter to inspect lowered operators, tensor shapes, quantized types,
graph splits, and optimization opportunities. You can also use these with MLIA.

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

## Note

Some artifacts are large, especially the OPT-125M `.pte` and `.etrecord` files.
They may take longer to download, load, and render than the smaller examples.

## License

This repository uses the Arm Education End User License Agreement for teaching
and learning content. See [LICENSE.md](LICENSE.md).
