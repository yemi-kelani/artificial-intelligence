import ConceptParser
import sys
from tokenizers import Tokenizer
from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
import numpy as np
from typing import NamedTuple

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

class Beam(NamedTuple):
    beam: list
    score: float
    terminated: bool
    length: int


def top_k(beam_width, probs):
    return torch.topk(probs, beam_width)


def top_p(beam_width, probs, nucleus):
    """
    This function was borrowed from transformers/generation_utils
    https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
    """
    logits = probs
    min_tokens_to_keep = beam_width

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(
        torch.nn.functional.softmax(sorted_logits, dim=-1),
        dim=-1
    )

    # Remove tokens with cumulative probability
    # above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > nucleus
    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[...,
                             1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float("Inf")
    print(logits)
    return logits

def get_candidates(
    beam_type, beam_width,
    probs, tokenizer,
    nucleus, exclude_tokens
):

    if beam_type == "top_k":
        scores, indicies = top_k(beam_width + len(exclude_tokens), probs)
        scores = scores.tolist()
    elif beam_type == "top_p":
        scores, indicies = top_p(
            beam_width + len(exclude_tokens), probs, nucleus)
        scores = scores.tolist()

    # omit tokens in exclude list
    candidates = []
    index, deleted_tokens = 0, 0
    num_indicies = indicies.size(dim=0)
    while index < num_indicies:
        token = tokenizer.decode(indicies[index])
        if token not in exclude_tokens:
            candidates.append(token)
        else:
            del scores[index - deleted_tokens]
            deleted_tokens += 1

        index += 1

    return scores[:beam_width], candidates[:beam_width]


def terminated(beams: list) -> bool:
    for beam in beams:
        if not beam.terminated:
            return False
    return True


def beam_search(
    start_state, model,
    tokenizer, parser=None,
    beam_type: str = "top_k",
    nucleus: float = 0.25,
    beam_width: int = 5,
    max_gen_length: int = 12,
    exclude_tokens: list = [" ", "\xa0", " \xa0"]
) -> Beam:
    """
    Beam Search will run until it hits an [END] token or until it 
    reaches its max_length.
    """

    model.to(device)
    model.eval()

    if isinstance(start_state, str):
        seed_text = start_state
    elif isinstance(start_state, list):
        seed_text = " ".join(start_state)
    else:
        raise ValueError(
            "start_state must be a string or list of tokens"
        )

    beams = [
        Beam([tokenizer.eos_token], 1.0, False, 0)
        for _ in range(beam_width)
    ]

    while not terminated(beams):
        new_beams = []

        # for each beam
        for i in range(len(beams)):

            # concatenate seed text with current beam
            # seed_text.extend(beams[i].beam)
            seed_text += " ".join(beams[i].beam)

            # get probability distribution over vocabulary
            with torch.no_grad():
                encoded_input = tokenizer(
                    seed_text, return_tensors="pt",
                    truncation=True, max_length=1024
                )

                # get all logits for next word
                logits = model(**encoded_input).logits[:, -1, :]

                # get probabilities for next word
                probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0)

            # get candidates according to decorder type
            if beam_type == "top_k" or beam_type == "top_p":
                scores, candidates = get_candidates(
                    beam_type, beam_width,
                    probs, tokenizer,
                    nucleus, exclude_tokens
                )
            else:
                raise TypeError(f"Invalid beam_type ' {beam_type} '.")

            # create a new beam for each candidate
            new_beams.extend([
                Beam(
                    # add candidate to beam
                    [*beams[i].beam, candidates[j]],
                    # multiply score by candidate score
                    beams[i].score * scores[j],
                    # done if eos_token or hit max length
                    candidates[j] == tokenizer.eos_token or
                    beams[i].length + 1 >= max_gen_length,
                    # increment beam length by 1
                    beams[i].length + 1
                )
                for j in range(len(candidates))
                # no duplicate tokens
                if candidates[j] != beams[i].beam[-1]
            ])

        # sort new beams descending
        if parser != None:
            new_beams.sort(
                key=lambda beam: beam.score +
                parser.score(" ".join(beam.beam)),
                reverse=True
            )  # add bonus parser score when sorting
        else:
            new_beams.sort(key=lambda beam: beam.score, reverse=True)

        del beams

        # select top new beams
        beams = new_beams[:beam_width]

    # return beam with the maximum score,
    # they should already be sorted unless
    # unless the beams are non-unique, in
    # which case it doesn't matter which
    return beams[0]


def beam2text(beam: Beam) -> str:
    return "".join(beam.beam[1:])


bos = '<|endoftext|>'
eos = '<|endoftext|>'
# pad = '<|pad|>'
special_tokens = {
    'eos_token': eos,
    'bos_token': bos,
    # 'pad_token': pad
}

tokenizer_orig = AutoTokenizer.from_pretrained("gpt2")  # transformer library
# with this, you don't have to manually define the new tokens' ids
tokenizer_orig.add_special_tokens(special_tokens)
tokenizer = Tokenizer.from_pretrained("gpt2")  # tokenizer library
# transformer library again but now with post processing
tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
num_added_toks = tokenizer.add_special_tokens(special_tokens)
print("NUM ADDED TOKENS:", num_added_toks)

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

cs = [
    {'start': 'Venus', 'end': 'a planet',
        'relationship': 'IsA', 'weight': 9.38083151964686},
    {'start': 'earth', 'end': 'planet',
        'relationship': 'RelatedTo', 'weight': 7.979724306014589},
    {'start': 'saturn', 'end': 'a planet', 'relationship': 'IsA', 'weight': 6.0},
    {'start': 'Mercury', 'end': 'the planet',
        'relationship': 'IsA', 'weight': 6.0},
    {'start': 'Uranus', 'end': 'a planet',
        'relationship': 'IsA', 'weight': 5.656854249492381},
    {'start': 'a planet', 'end': 'outer space',
        'relationship': 'AtLocation', 'weight': 5.656854249492381},
    {'start': 'Jupiter', 'end': 'a planet',
        'relationship': 'IsA', 'weight': 5.656854249492381},
    {'start': 'A planet', 'end': 'a large object',
        'relationship': 'IsA', 'weight': 5.291502622129181},
    {'start': 'a planet', 'end': 'orbit',
        'relationship': 'AtLocation', 'weight': 4.898979485566356},
    {'start': 'a planet', 'end': 'the universe',
        'relationship': 'AtLocation', 'weight': 4.898979485566356},
    {'start': 'a planet', 'end': 'live on',
        'relationship': 'UsedFor', 'weight': 4.47213595499958},
    {'start': 'Mars', 'end': 'a planet',
        'relationship': 'IsA', 'weight': 4.47213595499958},
    {'start': 'mercury', 'end': 'planet',
        'relationship': 'RelatedTo', 'weight': 4.125045454295019},
    {'start': 'Neptune', 'end': 'a planet', 'relationship': 'IsA', 'weight': 4.0},
    {'start': 'a sky', 'end': 'a planet',
        'relationship': 'AtLocation', 'weight': 2.82842712474619},
    {'start': 'moon', 'end': 'planet',
        'relationship': 'RelatedTo', 'weight': 2.551862065237853},
    {'start': 'planet', 'end': 'astronomy',
        'relationship': 'HasContext', 'weight': 2.0},
    {'start': 'superior planet', 'end': 'planet',
        'relationship': 'IsA', 'weight': 2.0}
]

sys.path.append("..")
parser = ConceptParser("../utilities.json")
knowledge = parser.concepts2paragraph(cs)

b = beam_search(
    start_state=knowledge+" What is the fourth planet from the sun?",
    model=model, tokenizer=tokenizer, parser=None, nucleus=0.3,
    beam_type="top_k"
)
print(beam2text(b))
