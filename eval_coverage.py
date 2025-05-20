import sys
import openai
from tqdm import tqdm
import time
import json
import argparse
import logging
import errno
import os
sys.path.append("./util/")
from azureopenai_api_call_vision import OpenAIAPI

# get azure openai key
from dotenv import load_dotenv
load_dotenv('./conf/gpt4o')
    
model = OpenAIAPI(loglevel="debug", model="gpt4o")

import base64
from mimetypes import guess_type
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=0.0000001)
parser.add_argument('--vqa-dir', type=str, default='sample_data')
parser.add_argument('--captions-file', type=str, default='sample_captions/llava1.6-vicuna_llama3_th1.0/captions_final.json')
args = parser.parse_args()

args.output_dir =  '/'.join(args.captions_file.split('/')[:-1])

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
mkdir(args.output_dir)
nn = args.captions_file.split('/')[-1].split('.')[0]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.output_dir, f'eval_recall_{nn}.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()
logging.info(args)

def a2l(answers):
    logging.info(answers)
    a = answers.replace('\n\n', '\n').strip().replace('. ', '\n')
    a = [b.split(')')[0] for b in a.split('\n')]
    logging.info(a[1::2])
    return a[1::2]

def generate_a_caption(questions, caption):
    questions_set = '\n\n'.join([f"{i}. {questions[i]}" for i in range(len(questions))])
    prompt_sys = f"Your role is to answer the given questions based on the provided caption. "
    prompt_sys += "I want to measure the amount of information in the caption. Therefore, if the correct answer to the question cannot be determined from the caption, you should answer it with \"I don't know\". "
    prompt_sys += "Do not use your own knowledge in your response. Do not use information that can be inferred from the question itself. Only use the information provided in the caption. "
    prompt_sys += "Answer the question by directly selecting the letter of the corresponding option. Do not repeat the question."
    user_input = f"Caption: {caption}.\n"
    user_input += f"Questions:\n{questions_set}" 
    messages = [
      {
        "role": "system",
        "content": [
          {
            "type": "text",
            "text": prompt_sys
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
              "type": "text",
              "text": user_input
          }
        ]
      }
    ]
    response = model.make_call(messages, args.temperature)
    if response == "":
        return ""
    elif response == "I don't know" or response == "I don't know.":
        return ["I don't know" for i in  range(len(questions))]
    out = a2l(response)
    return out

total_num = 0
corr = 0
skipped = []
with open(args.captions_file, 'r') as c:
    captions = json.load(c)
for i, (img_path, caption) in enumerate(captions.items()):
    name = os.path.splitext(img_path)[0]
    path = os.path.join(args.vqa_dir, f"{name}.json")
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            assert len(data["questions"]) == len(data["answers"])
        logging.info('='*10 + f' {path} ' + '='*10)
    except:
        continue
    # Question generation
    questions = data["questions"]
    is_continue = False
    num_try = 0
    while True:
        num_try += 1
        preds = generate_a_caption(questions, caption)
        if preds == "":
            skipped.append(name)
            is_continue = True
            break
        if len(preds) == len(data["answers"]) or num_try == 3:
            break
    if is_continue:
        continue
    preds = np.array([x.lower() for x in preds])
    answers = np.array([x.lower() for x in data["answers"]])
    corr += (preds == answers).sum()
    total_num += len(answers)
logging.info(f"Skipped: {skipped}")
logging.info(f"Recall: {corr / total_num}")
