## Week 07: Fine-tuning (PEFT & RLHF)

### Materials
- Lecture slides: [`./lecture.pdf`](./lecture.pdf)
- Video (Russian): [lecture](https://disk.yandex.ru/i/Axjf2u-hWDLAMw), [seminar](https://disk.yandex.ru/i/I_rAisZlK22FLA)
- From other courses (English): [EMNLP tutorial on PEFT](https://www.youtube.com/watch?v=KoOlcX3XLd4) (3.5h) | [MunichNLP short version](https://www.youtube.com/watch?v=StdrAJZsmw4) | [Hugging Face RLHF tutorial](https://www.youtube.com/watch?v=2MBJOuVq380)
- Optional: [lecture on task-driven chatbots](https://yadi.sk/i/4e_vqRDqwVGiFA) (Russian) | [lecture on conversation systems](https://disk.yandex.ru/i/XR1-8CghVOIK7A) (English)

### Practice
2 assignments, worth 5 points each (with bonus point opportunities)
- Seminar: [`./seminar.ipynb`](./seminar.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yandexdataschool/nlp_course/blob/2025/week07_finetuning/seminar.ipynb)
- Homework: [`./homework.ipynb`](./homework.ipynb) - includes optional hardcore RL fine-tuning assignment at the end

### Extra materials (RL finetuning & alignment)
- https://github.com/CarperAI/trlx - an alternative to trl designed for larger models
- A more detailed explanation of the reinforcement learning algorithms used in RLHF: [part 1](https://github.com/yandexdataschool/Practical_RL/tree/240c989fb1cca52effc289c4033438fa67af1e20/week06_policy_based) and [part 2](https://github.com/yandexdataschool/Practical_RL/tree/240c989fb1cca52effc289c4033438fa67af1e20/week09_policy_II)
- Antropic's take on aligning LLMs - [Constitutional AI](https://arxiv.org/abs/2212.08073)
- Earlier works on reinforcement learning for natural language generation:
  - [task-oriented conversation system](https://arxiv.org/abs/1703.07055)
  - [generating dialogues](https://arxiv.org/abs/1606.01541)
  - [sequential adversarial networks](https://arxiv.org/abs/1609.05473) (a.k.a. SeqGAN)
  - A large overview for machine translation (touching on RL, including RL failures) - [arxiv](https://arxiv.org/abs/1609.08144)
- as usual, there are dozens of links in the lecture slides (top of this readme)

### Extra materials (model architecture)
- "Building ML models like we build open-source software" by Colin Raffel - https://www.youtube.com/watch?v=0oGxT_i7nk8
- Rotary position embeddings explanation from EleutherAI - https://blog.eleuther.ai/rotary-embeddings/
- Group query attention to reduce the memory usage for inference - https://arxiv.org/abs/2305.13245v2
- Gated activations improve transformer (apparently due to divine benevolence) - https://arxiv.org/abs/2002.05202
- as usual, there are dozens of links in the lecture slides (top of this readme)
