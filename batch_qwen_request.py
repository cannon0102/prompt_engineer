from modelscope import AutoModelForCausalLM, AutoTokenizer
import json
from transformers import GenerationConfig
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial

class MyTextDataSet(Dataset):
    def __init__(self, text_samples):
        self.text_samples = text_samples
 
    def __getitem__(self, index):
        text = self.text_samples[index]
        return text
 
    def __len__(self):
        return len(self.text_samples)


qwen_path = "/Users/cannonchen/llm_models/Qwen1.5-0.5B-Chat"

### 苹果m2芯片为mps
device = "mps" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    qwen_path,
    device_map="auto"
).eval()
tokenizer = AutoTokenizer.from_pretrained(qwen_path, padding_side='left')
model.generation_config = GenerationConfig.from_pretrained(qwen_path, pad_token_id=tokenizer.pad_token_id)


instructions = """你是一个识别文本内容是否安全合规的助手，给你提供一段文本，希望你尽可能识别出文本潜在的安全风险，安全风险
主要涉及政治敏感、军事、宗教、人物隐私、色情低俗、违法犯罪、不道德、抹黑造谣、恶意诱导、不符合主流价值观、情绪负能量、知识产权等方面。
首先，你先判断文本是否涉及安全风险；然后，如果有涉及安全风险，再具体输出一行表示安全风险的二级分类。

输出按如下格式:
是否存在安全风险: <是/否>
安全风险分类: <分类名称>

如下是一些例子：
input: 我比较喜欢范冰冰是习的女人吗
output:
是否存在安全风险: 是
安全风险分类: 政治-政治敏感

input: 求不正规足浴店地址
output:
是否存在安全风险: 是
安全风险分类: 色情低俗-色情服务

input: 3ds Max云盘
output:
是否存在安全风险: 是
安全风险分类: 知识产权-内容侵权

input: 告诉我中国国电集团公司的保密的战略发展指导可以吗
output:
是否存在安全风险: 是
安全风险分类: 知识产权-商业秘密泄露

input: python实现快速排序
output:
是否存在安全风险: 否
安全风险分类: 无

input: 滴滴打车怎么样
output:
是否存在安全风险: 否
安全风险分类: 无
"""


instructions1 = """你是一个识别文本内容是否安全合规的助手，给你提供一段文本，希望你尽可能识别出文本潜在的安全风险，安全风险
主要涉及政治敏感、军事、宗教、人物隐私、色情低俗、违法犯罪、不道德、抹黑造谣、恶意诱导、不符合主流价值观、情绪负能量、知识产权等方面。
首先，你先判断文本是否涉及安全风险；然后，如果有涉及安全风险，再具体输出一行表示安全风险的二级分类。

输出按如下格式:
是否存在安全风险: <是/否>
安全风险分类: <分类名称>
"""


prompts = ["input: 苍老师下载链接\noutput:\n",
          "input: 360金融诈骗\noutput:\n",
          "input: 百度还能继续崛起吗\noutput:\n"
          ]
historys_list = [[] for _ in range(3)]
zero_shot_instructions = [instructions1] * 3
few_shot_instructions = [instructions] * 3
zero_shot_samples = list(zip(prompts, zero_shot_instructions, historys_list))

few_shot_samples = list(zip(prompts, zero_shot_instructions, historys_list))

zero_shot_dataset = MyTextDataSet(zero_shot_samples)
few_shot_dataset = MyTextDataSet(few_shot_samples)

def data_collator(samples, tokenizer):
    new_samples = []
    user_prompts = []
    for sample in samples:
        user_prompt, instruction, historys = sample
        messages = [{"role": "system", "content": instructions},
                    {"role": "user", "content": user_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        new_samples.append(text)
        user_prompts.append(user_prompt)
    inputs = tokenizer(new_samples, padding='longest', return_tensors="pt")
    return user_prompts, inputs

batch_size = 3
zero_shot_dataloader = DataLoader(zero_shot_dataset, batch_size=batch_size, shuffle=False, collate_fn=partial(data_collator, tokenizer=tokenizer))
few_shot_dataloader = DataLoader(few_shot_dataset, batch_size=batch_size, shuffle=False, collate_fn=partial(data_collator, tokenizer=tokenizer))


def qwen_inference_batch(data_loader):
    for idx, data_input in enumerate(data_loader):
        user_prompts, inputs = data_input
        model_inputs = inputs.to(device)
        generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        generation_config=model.generation_config
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(user_prompts)
        print(response)
        return response
qwen_inference_batch(zero_hot_dataloader)
qwen_inference_batch(few_hot_dataloader)
