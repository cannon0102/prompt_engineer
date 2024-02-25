from modelscope import AutoModelForCausalLM, AutoTokenizer
import json

qwen_path = ""
device = "mps" # the device to load the model onto
k = 3  # 采样结果数


tokenizer = AutoTokenizer.from_pretrained(qwen_path)
model = AutoModelForCausalLM.from_pretrained(
    qwen_path,
    device_map="auto"
).eval()

fake_data = {
    
}

instructions = """"""
user_prompt = """"""

messages = [
    {"role": "system", "content": instructions},
    {"role": "user", "content": user_prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)


generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    temperature=1,
    do_sample=True,
    num_return_sequences=k
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids.repeat(k,1), generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

def parse_score_from_review(review):
    try:
        score_line = review.split('\n')[-1]
        score_json = ':'.join(score_line.split(":")[1:]).strip()
        score_dict = json.loads(score_json)
        score0 = score_dict['answer0 score']
        score1 = score_dict['answer1 score']
        return [score0, score1]
    except Exception as e:
        print(f'Failed to parse scores from {review}')
        return [-1, -1]
   
for res in response:
    scores = parse_score_from_review(res)
    print(scores)