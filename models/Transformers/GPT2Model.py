import torch
from transformers import (GPT2Tokenizer,
                          GPT2LMHeadModel,
                          pipeline, set_seed,
                          PhrasalConstraint,
                          DisjunctiveConstraint)

class GPT2(torch.nn.Module):
    def __init__(
        self,
        model: str = "gpt2",
        seed: int = 0,
        max_gen_len: int = 20,
        num_returns: int = 1,
        num_beams: int = 5,
        unfrozen_threshold: int = 6
    ):
        super(GPT2, self).__init__()
        
        set_seed(seed)

        if model == "gpt2" or model == "gpt2-small":
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained(
                'gpt2',
                pad_token_id=self.tokenizer.eos_token_id
            )
            self.generator = pipeline('text-generation', model='gpt2')
        elif model == "gpt2-medium":
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.model = GPT2LMHeadModel.from_pretrained(
                'gpt2-medium',
                pad_token_id=self.tokenizer.eos_token_id
            )
            self.generator = pipeline('text-generation', model='gpt2-medium')
        elif model == "gpt2-large":
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
            self.model = GPT2LMHeadModel.from_pretrained(
                'gpt2-large',
                pad_token_id=self.tokenizer.eos_token_id
            )
            self.generator = pipeline('text-generation', model='gpt2-large')
        else:
            raise ValueError(
                f"Model type ' {model} ' not supported. [GPT2 __init__()]")
            
        self.tokenizer.add_special_tokens({
            'eos_token': '<|endoftext|>',
            'bos_token': '<|endoftext|>',
            'sep_token': '<|sep|>'
        })

        self.max_gen_len = max_gen_len
        self.beams = num_beams
        self.model_type = model
        self.num_returns = num_returns

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # freeze all be last few layers
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        for i, m in enumerate(self.model.transformer.h):        
            if i >= unfrozen_threshold:
                for parameter in m.parameters():
                    parameter.requires_grad = True 

        for parameter in self.model.transformer.ln_f.parameters():        
            parameter.requires_grad = True

        for parameter in self.model.lm_head.parameters():        
            parameter.requires_grad = True

    def forward(self, text):
        encoding = self.tokenizer(text, return_tensors="pt")
        encoding.to(self.device)

        output = self.model(**encoding)
        logits = output.logits[:, -1, :] # last hidden state

        return logits

    def decode(
        self, text: str,
        constrained: bool = False,
        concepts: list = []
    ) -> str:
        """
            GPT2.decode: Decodes inputs WITHOUT using fully connected
            classification head.
        """

        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            if self.generator:
                if not constrained:
                    return self.generator(
                        text, max_new_tokens=self.max_gen_len,
                        num_return_sequences=self.num_returns,
                        num_beams=self.beams
                    )[0]["generated_text"]

                print(
                    "Cannot perform constrained generation with generator. Generating manually.")

            inputs = self.tokenizer(text, return_tensors="pt")

            if constrained:
                tokenized_constraints = self.tokenizer(
                    concepts, add_special_tokens=False
                ).input_ids
                constraints = DisjunctiveConstraint(list(tokenized_constraints))
                output = self.model.generate(
                    inputs["input_ids"],
                    constraints=[constraints],
                    max_new_tokens=self.max_gen_len,
                    num_beams=self.beams,
                    num_return_sequences=self.num_returns,
                    no_repeat_ngram_size=1,
                    remove_invalid_values=True,
                ) 
            else:
                output = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=self.max_gen_len,
                    num_beams=self.beams
                )

            output_text = self.tokenizer.decode(
                output[0], skip_special_tokens=True)
            return output_text


# lm = GPT2(model="gpt2-medium", max_gen_len=2)
# import numpy as np
# print(np.shape(lm.forward("What is the third planet from the sun?")))
# print(lm.decode("What is the third planet from the sun?",
#       concepts=["have fun"],
#       constrained=True))
