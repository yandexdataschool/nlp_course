### Large language models


- Slides: [lecture_llm.pdf](./lecture_llm.pdf)
- Practice session: [practice.ipynb](./practice.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yandexdataschool/nlp_course/blob/fall22/week08_llm/practice.ipynb)
- Video (in russian): [lecture](https://disk.yandex.ru/i/YCRr1gRuzXpZJA), [practice](https://disk.yandex.ru/i/1HaYtOxWZlHB5g)


### Read more

* Slides ([lecture_llm.pdf](./lecture_llm.pdf)) contain many links for further reading
* How post-training quantization works: https://arxiv.org/abs/2208.07339 
* An overview of running large models: https://huggingface.co/docs/accelerate/package_reference/big_modeling 
* A general library for different adapter types: https://adapterhub.ml/


### [practice notes] Fine-grained inference

If for some reason you're not satisfied with `model.generate` interface, you can write your own inference code with iterative forward passes. Here's how it's done:
```python
prefix = "Mark Zuckerberg is"  # same as above
batch = tokenizer(prefix, return_tensors='pt')
past_key_values = None
with torch.cuda.amp.autocast():
  for i in range(50):
    outputs = model.forward(**batch, use_cache=True, past_key_values=past_key_values)
    probs = outputs.logits[0, -1].div(0.8).softmax(-1)
    token = torch.multinomial(probs, 1).view([])

    print(tokenizer.decode(token), end=' ', flush=True)
    past_key_values = outputs.past_key_values
    batch = dict(input_ids=outputs.logits[0, -1].argmax(-1).reshape(1, 1),
                 attention_mask=torch.ones(1, past_key_values[0][0].shape[-2] + 1, device='cuda'))
```


### [practice notes] How to optimize for inference

The code below converts training-optimized 8bit weights into inference-optimized layout. It should result in significantly faster inference in the same memory footprint. 
However, if you do this, you can no longer run training --
 there is no way to un-convert after the first optimized forward!

```python
model.config.use_cache = True
for module in model.modules():
    if isinstance(module, bnb.nn.Linear8bitLt):
        module.state.memory_efficient_backward = False
```

