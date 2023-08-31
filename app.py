import os
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse

model_name_or_path = "TheBloke/vicuna-7B-v1.3-GPTQ"
model_basename = "vicuna-7b-v1.3-GPTQ-4bit-128g.no-act.order"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class InferlessPythonModel:
    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
            use_safetensors=True,
            trust_remote_code=True,
            device_map='auto',
            use_triton=True,
            quantize_config=None
        )


    def infer(self, inputs):
        prompt = inputs["prompt"]
        prompt_template=f'''USER: {prompt}
        ASSISTANT:'''

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        generated_text = pipe(prompt_template)[0]['generated_text']
        return {"generated_text": generated_text}

    def finalize(self):
        self.tokenizer = None
        self.model = None
