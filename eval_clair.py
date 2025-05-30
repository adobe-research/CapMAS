import json
import errno
import os
import logging
import re
import sys
from typing import List, Optional, Tuple
import argparse
sys.path.append("./util/")
from azureopenai_api_call_vision import OpenAIAPI
from dotenv import load_dotenv
load_dotenv('./conf/gpt4o')
model = OpenAIAPI(loglevel="debug", model="gpt4o")

_CLAIR_PROMPT = """\
You are trying to tell if a candidate set of captions is describing the same image as a reference set of captions.
Candidate set:
{candidate_statements}
Reference set:
{target_statements}
On a precise scale from 0 to 100, how likely is it that the candidate set is \
describing the same image as the reference set? (JSON format, with a key "score", \
value between 0 and 100, and a key "reason" with a string value.)
"""

def clair(
    candidates: List[str],
    targets: List[str],
    max_tokens: int = 1024,
) -> Tuple[float, Optional[str]]:
    # Compute the CLAIR score for a list of candidates and targets.

    # Format the canndidates and targets
    candidate_statements = [f"- {c}\n" for c in candidates]
    target_statements = [f"- {t}\n" for t in targets]
    formatted_prompt = _CLAIR_PROMPT.format(
        candidate_statements="".join(candidate_statements),
        target_statements="".join(target_statements),
    )

    temperature, score, reason = 0.0, None, None
    for _ in range(3):
        # Run the model
        logging.info(f'CLAIR prompt: "{formatted_prompt}"')
        messages = [
          {
            "role": "system",
            "content": [
              {
                "type": "text",
                "text": formatted_prompt
              }
            ]
          },
        ]
        response = model.make_call(messages, temperature)
        logging.info(f'CLAIR response: "{response.strip()}"')

        # Parse the first JSON object in the response
        try:
            parsed = response.split("{")[1]
            parsed = "{" + parsed.split("}")[0] + "}"
            data = json.loads(parsed)
            score = float(data["score"])
            reason = data.get("reason", 'Unknown')
            break
        except (json.JSONDecodeError, KeyError, IndexError):
            # Try to extract the first number in the response using regex
            parsed = re.findall(r"\d*\.?\d+", response)
            if len(parsed) > 0:
                score = float(parsed[0])
                if score < 1:
                    score *= 100 # This is a weird situation where some models auto-normalize the score for us.

                # Look for the word "reason" in the response, and extract anything after it (ignoring case)
                reason = re.findall(r"(?i)reason.*", response)
                if len(reason) > 0:
                    # Clean up the reason a bit.
                    reason = reason[0].strip()[len('reason'):].replace(':', '').strip()
                else:
                    reason = 'Unknown'
                break
            else:
                logging.warn(
                    f"Could not parse response from CLAIR: {response}. Retrying"
                )
                continue
    else:
        logging.error("Could not parse response from CLAIR after 3 tries. Setting score to 0.")
        score = 0.0
        reason = None

    return score / 100, reason

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--captions-file', type=str, default='sample_captions/llava1.6-vicuna_llama3_th1.0/captions_final.json')
    args = parser.parse_args()
    
    args.output_dir =  '/'.join(args.captions_file.split('/')[:-1])
    log_name = args.captions_file.split('/')[-1]
    log_name = log_name.split('.')[0]
    mkdir(args.output_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, f'eval_clair_{log_name}.log')),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()
    logging.info(args)

    with open(args.captions_file, 'r') as c:
        captions = json.load(c)
    with open('data/data.jsonl', 'r') as json_file:
        json_list = list(json_file)

    skipped = []
    score_accu = 0.
    for i, json_str in enumerate(json_list):
        data = json.loads(json_str)
        try:
            pred = captions[data["image"]+".jpg"]
            ref = data["caption"]
            score, reason = clair([pred], [ref], max_tokens=128)
            score_accu += score
            if reason == None:
                skipped.append(data["image"])
        except KeyError:
            skipped.append(data["image"])
        except Exception as e:
            print(e)
            sys.exit(1)

    logging.info(f'SKIP: {skipped}')
    logging.info(f'CLAIR: {score_accu / (len(json_list) - len(skipped))}')
