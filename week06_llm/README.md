### Large language models


- [Lecture slides](https://drive.google.com/file/d/1IOx71suOn8uF_AbNrPhQxjnNNA5UGQY1/view?usp=share_link) 
- Video (in russian): [lecture](https://disk.yandex.ru/i/O1oEoThF0h02GA), [practice](https://disk.yandex.ru/d/A-giSxCD1ydPzA)
- Practice session: [practice.ipynb](./practice.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yandexdataschool/nlp_course/blob/2023/week06_llm/practice.ipynb)


Applications mentioned in the lecture:
- [the Gandalf game](https://gandalf.lakera.ai/) by Lakera.ai
- [aidungeon.com](https://play.aidungeon.com/) - an LLM-generated role-playing game
- [ora.ai](https://ora.ai) - a tool to build your own chatbot with prompting
- this is not an exhaustive list: there's a million of various applications using LLMs

Open-source models mentioned in the lecture:
- LLaMA-2 - https://huggingface.co/meta-llama/Llama-2-70b
- Falcon - https://huggingface.co/tiiuae/falcon-180B
- BLOOM - https://huggingface.co/bigscience/bloom

Some of those models require you to apply for access, and model authors may take time to process your application. While you are waiting for your license to be processed, you may wish to browse the huggingface hub for alternative (e.g. quantized) versions of the same model that are available immediately with no application. For example, here's a [LLaMA-2-70B](https://huggingface.co/TheBloke/Llama-2-70B-Chat-GPTQ) quantized to 4-bit and available using the same `transformers.AutoModelForCausalLM.from_pretrained` syntax. Please note that, while the hub allows you to download and use those model versions without officially applying for access to LLaMA-2, the original model's license may restrict this kind of usage.


Extra materials:
- Glitch tokens (lecture mentions SolidGoldMagikarp) - [blog post by Jessica Rumbelow, mwatkins](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)
- ["Sparks of AGI"](https://arxiv.org/abs/2303.12712) - a controversial but influential paper about worrying LLM abilities
- [BigBench](https://github.com/google/BIG-bench) - a benchmark of emergent LLM abilities mentioned in the slides
- Chain of thought papers: Few-shot: [Wei et al. (2022) few-shot](https://arxiv.org/abs/2201.11903)
- A guide to prompt injection and jailbreaking: https://learnprompting.org/docs/prompt_hacking/injection
- A repo with popular jailbreaks for GPTx models: https://github.com/0xk1h0/ChatGPT_DAN
- A ton of other cool stuff linked in the lecture slides (see the top of this readme)

