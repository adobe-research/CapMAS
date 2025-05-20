# [ICML 2025] Toward Robust Hyper-Detailed Image Captioning: A Multiagent Approach and Dual Evaluation Metrics for Factuality and Coverage

This repository contains the evaluation code and data of our ICML 2025 paper, [Toward Robust Hyper-Detailed Image Captioning: A Multiagent Approach and Dual Evaluation Metrics for Factuality and Coverage](https://arxiv.org/abs/2412.15484).

## Prerequisites
### Packages
* openai>=1.14.1
* python-dotenv==1.0.1

### Dataset download
```dataset download
from huggingface_hub import hf_hub_download

local_path = hf_hub_download(
    repo_id="saehyungl/CapMAS",
    filename="images_capmas.tar.gz",
    repo_type="dataset"
)
print("Downloaded to:", local_path)
```
Or you can download it using [this URL](https://huggingface.co/datasets/saehyungl/CapMAS/resolve/main/images_capmas.tar.gz?download=true).
Our evaluation uses a subset of the [DOCCI](https://google.github.io/docci/) images.

## Captioning
Please generate captions for the 1,000 downloaded images for captioning evaluation.
Summarize the generated captions into a dictionary where the key is the corresponding image file name, and save it as a .json file.
```captions file
{
    "aar_test_04600.jpg": <caption_aar_test_04600>,
    "aar_test_04601.jpg": <caption_aar_test_04601>,
    ...
    "test_00599.json": <caption_test_00599>,
}
```
You may refer to the [sample captions](https://github.com/david-yoon/CapMAS_private/blob/main/sample_captions/llava1.6-vicuna_llama3_th1.0/captions_final.json) for guidance.

## Evaluation
We provide the evaluation codes for the three metrics used in our paper: **Factuality**, **Coverage**, and **CLAIR** (Chan et al., EMNLP 2023). These evaluations rely on GPT-4o, so please fill in your OpenAI API key **OR** Azure OpenAI credentials in the `conf/gpt4o` file.
### Factuality (ours)
```factuality
python eval_factuality.py --image-dir <the image directory path> --captions-file <the caption .json file path>
```
### Coverage (ours)
```coverage
python eval_coverage.py --vqa-dir data/COVERAGE_TEST_VQA --captions-file <the caption .json file path>
```
### CLAIR
```clair
python eval_clair.py --captions-file <the caption .json file path>
```

## References
1. [DOCCI (Onoe et al., ECCV 2024)](https://google.github.io/docci/#downloads)
2. [ImageInWords (Garg et al., EMNLP 2024)](https://github.com/google/imageinwords)
3. [CLAIR (Chan et al., EMNLP 2023)](https://github.com/davidmchan/clair)




## Cite
If you use the **CapMAS** dataset, filtering pipeline, or code from this repository, please cite the [paper](https://arxiv.org/pdf/2412.15484):

```bibtex
@article{lee2024toward,
  title={Toward Robust Hyper-Detailed Image Captioning: A Multiagent Approach and Dual Evaluation Metrics for Factuality and Coverage},
  author={Lee, Saehyung and Yoon, Seunghyun and Bui, Trung and Shi, Jing and Yoon, Sungroh},
  journal={arXiv e-prints},
  pages={arXiv--2412},
  year={2024}
}
```

## License

The evaluation code and needle set data is licensed under the [Adobe Research License](LICENSE). The license prohibits commercial use and allows non-commercial research use.
