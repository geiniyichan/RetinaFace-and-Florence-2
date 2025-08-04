from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# 加载模型和处理器
model_dir = "Florence-2-large-ft"  # 替换为你的模型路径
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

# 加载本地图片
image_path = "result/20_Family_Group_Family_Group_20_15.jpg"  # 替换为你的图片路径
image = Image.open(image_path)

# 构建任务提示，修改为More Detailed Caption任务
task_prompt = "<DETAILED_CAPTION>"

# 处理输入并生成描述
inputs = processor(text=task_prompt, images=image, return_tensors="pt")
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=3,
    early_stopping=False,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

# 输出结果
print("Generated Description:")
print(parsed_answer)