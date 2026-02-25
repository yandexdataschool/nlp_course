# YSDA Natural Language Processing course
* This is the 2025 iteration of the course, materials are added as we prepare them.
* Lecture and seminar materials for each week are in ./week* folders, see README.md for materials and instructions
* Any technical issues, ideas, bugs in course materials, contribution ideas - add an [issue](https://github.com/yandexdataschool/nlp_course/issues)
* Installing libraries and troubleshooting: [this thread](https://github.com/yandexdataschool/nlp_course/issues/1).


# Syllabus
- [__week01__](./week01_embeddings) __Word Embeddings__
  - Lecture: Word embeddings. Distributional semantics. Count-based (pre-neural) methods. Word2Vec: learn vectors. GloVe: count, then learn. Evaluation: intrinsic vs extrinsic. Analysis and Interpretability. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_word_emb)
  - Seminar: Playing with word and sentence embeddings
  - Homework: Embedding-based machine translation system

- [__week02__](./week02_lm) __Language Modeling__
  - Lecture: Language Modeling: what does it mean? Left-to-right framework. N-gram language models. Neural Language Models: General View, Recurrent Models, Convolutional Models. Evaluation. Practical Tips: Weight Tying. Analysis and Interpretability. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_lang_models)
  - Seminar: Build a N-gram language model from scratch
  - Homework: Neural LMs & smoothing in count-based models.

- [__week03__](./week03_attention) __Seq2seq and Attention__
  - Lecture: Seq2seq Basics: Encoder-Decoder framework, Training, Simple Models, Inference (e.g., beam search). Attention: general, score functions, models. Transformer: self-attention, masked self-attention, multi-head attention; model architecture. Subword Segmentation (BPE). Analysis and Interpretability: functions of attention heads; probing for linguistic structure. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_seq2seq_attn)
  - Seminar: Basic sequence to sequence model
  - Homework: Machine translation with attention

- [__week04__](./week04_transfer) __Transfer Learning__
  - Lecture: What is Transfer Learning? Great idea 1: From Words to Words-in-Context (CoVe, ELMo). Great idea 2: From Replacing Embeddings to Replacing Models (GPT, BERT). (A Bit of) Adaptors. Analysis and Interpretability. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_transfer)
  - Homework: fine-tuning a pre-trained BERT model

- [__week05__](./week05_llm) __Large Language Models__
  - Lecture: Scaling laws. Emergent abilities. Open-source LLMs.
  - Practice: hands-on with open-source LLMs

- [__week06__](./week06_prompting) __Prompting & In-Context Learning__
  - Lecture: Prompting techniques. Chain-of-Thought reasoning. In-context learning: how and why it works. Analysis and Interpretability.
  - Homework: manual prompt engineering and chain-of-thought reasoning

- [__week07__](./week07_finetuning) __Fine-tuning (PEFT & RLHF)__
  - Lecture: Parameter-efficient fine-tuning (LoRA, adapters). Reinforcement Learning from Human Feedback (RLHF).
  - Seminar + Homework

- [__week08__](./week08_efficiency) __Efficiency__
  - Lecture: Quantization. Distillation. Pruning. Speculative decoding.
  - Homework

- [__week09__](./week09_retrieval) __Retrieval-Augmented Generation (RAG)__
  - Lecture: Dense retrieval. RAG architectures.
  - Practice

- [__week10__](./week10_agents) __AI Agents__
  - Lecture: Agent architectures. Tool use. Memory.
  - Seminar + Homework

- [__week11__](./week11_interpretability) __Interpretability__
  - Lecture: Probing. Mechanistic interpretability.
  - Seminar + Homework

- [__week12__](./week12_multimodal) __Multimodal LLMs__

- [__week13__](./week13_llm_systems) __Building LLM Systems__

- [__week14__](./week14_agents_production) __AI Agents in Production__

# Contributors & course staff
Course materials and teaching performed by
- [Elena Voita](https://lena-voita.github.io) - original course author
- [Michael Diskin](https://diskin.me) - responsible for the 2025 edition; lecturer
- [Just Heuristic](https://github.com/justheuristic) - most of the seminars and some lectures
- [Ignat Romanov](https://github.com/RomanovIgnat), [George Yakushev](https://github.com/Mr-DarkTesla), [Andrei Panferov](https://github.com/BlackSamorez) - lectures
- [Natasha Badanina](https://github.com/dokapoka) - course admin for on-campus students
- [Boris Kovarsky](https://github.com/kovarsky), [David Talbot](https://github.com/drt7), [Sergey Gubanov](https://github.com/esgv), Ruslan Svirschevski - help build course materials and/or held some classes
- [30+ volunteers](https://github.com/yandexdataschool/nlp_course/graphs/contributors) who contributed and refined the notebooks and course materials. Without their help, the course would not be what it is today
- [A mighty host of TAs](https://lk.yandexdataschool.ru/courses/2023-autumn/7.1171-avtomaticheskaia-obrabotka-tekstov/) who stoically grade hundreds of homework submissions from on-campus students each year
