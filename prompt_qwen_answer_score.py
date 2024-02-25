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


instructions = """你是一个标注回答偏好和安全性的助手， 现在给你一个问题与两个答案，希望收到你对这两个答案在解决上述问题的表现的反馈。请从答案的有用性、相关性、准确性和详细程度进行评分。
每个答案都会在一个1到10的评分尺度上获得一个总体评分，其中更高的评分表示总体表现更好。请首先提供一个全面的评价解释，避免任何潜在的偏见，并确保回答的呈现顺序不会影响您的判断。然后，输出一行表示答案0和答案1的评分。"""
user_prompt = """input:
question:保持健康的三个提示。
answer0:

健康饮食：确保你吃各种食物，包括水果、蔬菜、全谷类食品和瘦肉蛋白。避免吃垃圾食品，如快餐和含糖饮料。

定期锻炼：每天进行30分钟的有氧运动，如快走、慢跑或游泳。

保持积极的态度：保持乐观和积极的态度，可以帮助你应对生活中的挑战。

记住，保持健康需要时间和耐心。不要急于求成，要循序渐进地改变你的生活方式。
answer1:保持健康的三个提示是：

均衡饮食：确保你的饮食中含有各种营养丰富的食物，包括蛋白质、脂肪、碳水化合物和各类蔬菜、水果和坚果。

适度运动：每天进行至少30分钟的中等强度的运动，如步行、跑步、游泳、瑜伽等。

保持健康的心态：保持积极的态度，学会放松和减轻压力，这些都对保持健康起着关键的作用。


Output With the following format:
评价解释: <评估解释>
两个回答评分: {"answer0 score": <score>, "answer1 score": <score>}"""


评价解释：根据用户提供的问题，两个答案都给出了有效的建议来保持健康。答案0的回答强调了健康饮食、定期锻炼和保持积极的态度，但没有提供具体的建议。答案1的回答则给出了更具体的建议，如均衡饮食、适度运动和保持健康心态。因此，答案1获得了更高的评分。
两个回答评分: {"answer0 score": 7, "answer1 score": 9}



你是一个标注回答偏好和安全性的助手， 现在给你一个问题与两个不同的答案，希望收到你对这两个答案在解决上述问题的表现的反馈。请从答案的有用性、相关性、准确性和详细程度进行评分。每个答案都会在一个1到10的评分尺度上获得一个总体评分，其中更高的评分表示总体表现更好。直接输出一行表示答案0和答案1的评分。
输出需遵从如下格式:
两个回答评分: {"answer0 score": <score>, "answer1 score": <score>}
