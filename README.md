# Code-DFQ-SAM (for review)

Here is the source code to reproduce DFQ-SAM. 

**Please note that the code is only used for review purposes and has not been publicly released.**

## Environment

- The GPU is recommended to be a single NVIDIA A6000.
- Representative package: python=3.10, torch=2.0.1 (In our experiments, DFQ-SAM has good compatibility with versions of other packages.)

## Data-free Quantization for SAM
First, download the pre-trained model checkpoint from [Model](https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view?usp=drive_link).

Then, you can run the following command to quantize SAM via the proposed DFQ-SAM.
```bash
export CUDA_VISIBLE_DEVICES=0
cd ./DFQ-SAM
python test_quant.py
```
- data_path: Path to test dataset.
- sam_checkpoint: Path to SAM checkpoint.
- batch_size: 1.
- image_size: Default value is 256.
- boxes_prompt: Use Bbox prompt to get segmentation results. 
- point_num: Specifies the number of points. Default value is 1.
- iter_point: Specifies the number of iterations for point prompts.
- encoder_adapter: Set to True if using SAM-Med2D's pretrained weights.
- save_pred: Whether to save the prediction results.


## Acknowledgments
The code of DFQ-SAM is based on [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D). We thank for their open-sourced code.