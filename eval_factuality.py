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
parser.add_argument('--image-dir', type=str, default='sample_data')
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.output_dir, 'eval_precision.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()
logging.info(args)

# Function to encode the image
def encode_image(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def generate_q(caption):
    prompt_sys = f"I want to verify if the given CAPTION is accurate. To assist with this verification, decompose the given CAPTION into atomic propositions. All parts of the caption must be broken down into propositions. The outputs should follow the following format:'1. proposition one\n2. proposition two\n3. proposition three'"
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
            "text": caption,
            
          }
        ]
      }
    ]
    response = model.make_call(messages, args.temperature)
    if response == "":
        return ""
    out = response.replace('\n\n', '\n').split('\n')
    atoms = []
    for o in out:
        if o[0].isdigit() and len(o.split(". ")) > 1:
            atoms.append(o)
    return '\n'.join(atoms)

def a2l(answers):
    a = answers.replace('\n\n', '\n').strip().replace('. ', '\n')
    a = [b.split(')')[0] for b in a.split('\n')]
    a = [b.split()[0] for b in a]
    return a[1::2]

def generate_a_image(questions, path, gt_caption):
    num_questions = len(questions.split("\n"))
    image_url = encode_image(path)
    prompt_sys = "Your role is to determine whether the given propositions are True or False based on the provided image and its description."
    prompt_sys += "The outputs should follow the following format:'1. True/False\n2. True/False\n3. True/False\n4. True/False\n5. True/False\n ...'."
    prompt_sys += "The number of True/False answers must match the number of propositions."
    user_input = f"Description: {gt_caption}\n\nPropositions:\n{questions}"
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
          },
          {
            "type": "image_url",
            "image_url": {
              "url": image_url
            }
          }
        ]
      }
    ]
    num_try = 0
    retry = False
    while True:
        num_try += 1
        response = model.make_call(messages, args.temperature)
        out = a2l(response)
        if len(out) == num_questions:
            break
        elif num_try == 3:
            logging.info("The number of answers does not match the number of questions.")
            logging.info("Let's retry from the decomposition.")
            logging.info(questions)
            logging.info(out)
            retry = True
    return out, retry

total_num = 0
corr = 0
fmt="\n"
with open(args.captions_file, 'r') as c:
    captions = json.load(c)
with open('data/data.jsonl', 'r') as json_file:
    json_list = list(json_file)
skipped = []
for i, json_str in enumerate(json_list):
    data = json.loads(json_str)
    name = data["image"]
    path = os.path.join(args.image_dir, name+".jpg")
    if name+".jpg" not in captions:
        skipped.append(name)
        continue
    logging.info('='*10 + f' {path} ' + '='*10)
    outputs = []
    # Question generation
    while True:
        is_continue = False
        questions = generate_q(captions[name+".jpg"])
        if questions == "":
            skipped.append(name)
            is_continue = True
            break
        image_answers, retry = generate_a_image(questions, path, data["caption"])
        if retry == False:
            break
    if is_continue:
        continue
    logging.info(captions[name+".jpg"])
    corr_prev = corr
    for j, q in enumerate(questions.split(fmt)):
        output = {}
        output['question'] = ' '.join(q.split(' ')[1:])
        output['caption_answer'] = "True"
        output['image_answer'] = image_answers[j]
        logging.info(output['question'])
        logging.info(output['caption_answer'])
        logging.info(output['image_answer'])
        outputs.append(output)
        total_num += 1
        corr += image_answers[j] == output['caption_answer']
logging.info(f"Skipped: {skipped}")
logging.info(f"Precision: {corr / total_num}")
