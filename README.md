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
You may refer to the [sample captions](https://github.com/adobe-research/CapMAS/blob/master/sample_captions/llava1.6-vicuna_llama3_th1.0/captions_final.json) for guidance.

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

## Benchmarking
To evaluate image captions in our paper, we used the following method:
1. We collect model responses for the following five prompts:
```prompts
queries = [
    "Describe the given image in a very detailed manner.",
    "Provide a detailed description of the specified image.",
    "Elaborate on the details of the image provided.",
    "Offer an in-depth description of the given image.",
    "Thoroughly describe the features of the specified image.",
    ]

```
2. We use [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) to merge the five captions into a single caption. The following prompt is used for this step:
```summarize
SYSTEM: "This is a hard problem. Carefully summarize in ONE detailed caption based on the following 5 captions by different people describing the same image. Be sure to describe everything, and avoid hallucination. Provide the detailed caption in the format '### {Detailed caption} ###'."
USER: f"Caption 1: {generated_captions[0]}
        Caption 2: {generated_captions[1]}
        Caption 3: {generated_captions[2]}
        Caption 4: {generated_captions[3]}
        Caption 5: {generated_captions[4]}"
```
3. We then evaluate the resulting caption once. We found that this approach is relatively robust and cost-efficient.

### Results
| Method         | CLAIR | Factuality | Coverage | Avg. |
|---------------|--------|-------|---------|------|
| LLaVA-v1.5-7B | 62.1 | 52.8 | 40.3 | 51.7|
| [VCD (Leng et al., 2024)](https://github.com/DAMO-NLP-SG/VCD)   |59.7  |	44.6  |	46.2  |	50.2  |
| [OPERA (Huang et al., 2024)](https://github.com/shikiw/OPERA )   |59.1 |	53.0 |	40.7 |	50.9 |
| [SPARC (Jung et al., 2025)](https://github.com/mingi000508/SPARC)| 64.7|	50.2	|44.3	|53.1|
| [LRV (Liu et al., 2024)](https://github.com/FuxiaoLiu/LRV-Instruction)|39.7|	29.1|	43.7|	37.5|
| [LURE (Zhou et al., 2024)](https://github.com/YiyangZhou/LURE)|	57.2|	51.9|	31.2|	46.8|
| [Volcano (Lee et al., 2024)](https://github.com/kaistAI/Volcano)|	63.9|	53.7|	44.1|	53.9|

*All methods, except for LRV and LURE, use LLaVA-v1.5-7B, while the LRV and LURE methods employ the MiniGPT-4 model as provided by their respective authors.  
*The difference in Coverage scores compared to the paper is because we conducted an additional round of review and refinement on the VQA samples after the paper was written.  
*These results are based solely on the IIW-400 dataset (aar_test_04600.jpg – aar_test_04999).

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
