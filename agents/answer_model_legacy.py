# Qwen3-4B in action.
import time
import torch
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.random.manual_seed(0)
class AAgent(object):
    def __init__(self, **kwargs):
        # self.model_type = input("Available models: Qwen3-1.7B and Qwen3-4B. Please enter 1.7B or 4B: ").strip()
        self.model_type = kwargs.get('model_type', '4B').strip()
        # model_name = "Qwen/Qwen3-4B"
        model_name = "/jupyter-tutorial/hf_models/Qwen3-4B"
        
        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate_response(self, message: str|List[str], system_prompt: Optional[str] = None, **kwargs)->str:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        if isinstance(message, str):
            message = [message]
        # Prepare all messages for batch processing
        all_messages = []
        for msg in message:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg}
            ]
            all_messages.append(messages)
        
        # convert all messages to text format
        texts = []
        for messages in all_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            texts.append(text)

        # tokenize all texts together with padding
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        tgps_show_var = kwargs.get('tgps_show', False)
        # conduct batch text completion
        if tgps_show_var: start_time = time.time()   
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=kwargs.get('max_new_tokens', 1024),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if tgps_show_var: generation_time = time.time() - start_time

        # decode the batch
        batch_outs = []
        if tgps_show_var: token_len = 0
        for i, (input_ids, generated_sequence) in enumerate(zip(model_inputs.input_ids, generated_ids)):
            # extract only the newly generated tokens
            output_ids = generated_sequence[len(input_ids):].tolist()
            
            # compute total tokens generated
            if tgps_show_var: token_len += len(output_ids)

            # remove thinking content using regex
            # result = re.sub(r'<think>[\s\S]*?</think>', '', full_result, flags=re.DOTALL).strip()
            index = len(output_ids) - output_ids[::-1].index(151668) if 151668 in output_ids else 0
            
            # decode the full result
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            batch_outs.append(content)
        if tgps_show_var:
            return batch_outs[0] if len(batch_outs) == 1 else batch_outs, token_len, generation_time
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None
        
if __name__ == "__main__":
    # Single message (backward compatible)
    ans_agent = AAgent()
    response, tl, gt = ans_agent.generate_response("Solve: 2x + 5 = 15", system_prompt="You are a math tutor.", tgps_show=True, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True)
    print(f"Single response: {response}")
    print(f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl/gt:.2f}")
    print("-----------------------------------------------------------")
          
    # Batch processing (new capability)
    messages = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, gt = ans_agent.generate_response(messages, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True, tgps_show=True)
    print("Responses:")
    for i, resp in enumerate(responses):
        print(f"Message {i+1}: {resp}")
    print(f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl/gt:.2f}")
    print("-----------------------------------------------------------")

    # Custom parameters
    response = ans_agent.generate_response(
        "Write a story", 
        temperature=0.8, 
        max_new_tokens=512
    )
    print(f"Custom response: {response}")
