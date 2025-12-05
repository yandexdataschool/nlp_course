### Large language models


- [Lecture slides](https://docs.google.com/presentation/d/1kqtAPArk481cDLgSA_Mt0yUbypzDJLMQ/edit?usp=sharing&ouid=114346045969351679204&rtpof=true&sd=true) 
- Video (in russian): [lecture & practice](https://disk.yandex.ru/i/hv66j4yuYZWlBw)
- Practice session: [seminar.ipynb](./seminar.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yandexdataschool/nlp_course/blob/2025/week05_llm/seminar.ipynb)
  <!-- - Optional demo for combining LLMs and web search via HF Agents [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yandexdataschool/nlp_course/blob/2024/week06_llm/demo_agents.ipynb) -->


Open-source models mentioned in the lecture:
- Qwen - https://huggingface.co/Qwen
- LLaMA - https://huggingface.co/meta-llama/
- Falcon - https://huggingface.co/tiiuae/falcon-180B
- BLOOM - https://huggingface.co/bigscience/bloom

Some of those models require you to apply for access, and model authors may take time to process your application. While you are waiting for your license to be processed, you may wish to browse the huggingface hub for alternative (e.g. quantized) versions of the same model that are available immediately with no application. For example, here's a [LLaMA-2-70B](https://huggingface.co/TheBloke/Llama-2-70B-Chat-GPTQ) quantized to 4-bit and available using the same `transformers.AutoModelForCausalLM.from_pretrained` syntax. Please note that, while the hub allows you to download and use those model versions without officially applying for access to LLaMA-2, the original model's license may restrict this kind of usage.


Extra materials:
- Glitch tokens (lecture mentions SolidGoldMagikarp) - [blog post by Jessica Rumbelow, mwatkins](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)
- ["Sparks of AGI"](https://arxiv.org/abs/2303.12712) - a controversial but influential paper about worrying LLM abilities
- [BigBench](https://github.com/google/BIG-bench) - a benchmark of emergent LLM abilities mentioned in the slides
