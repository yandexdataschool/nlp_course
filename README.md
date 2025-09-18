# YSDA Natural Language Processing course
* This is the 2025 iteration of the course, materials are added as we prepare them. **For full 2025 course materials, go to [this branch](https://github.com/yandexdataschool/nlp_course/tree/2024)**
* Lecture and seminar materials for each week are in ./week* folders, see README.md for materials and instructions
* Any technical issues, ideas, bugs in course materials, contribution ideas - add an [issue](https://github.com/yandexdataschool/nlp_course/issues)
* Installing libraries and troubleshooting: [this thread](https://github.com/yandexdataschool/nlp_course/issues/1).


# Syllabus
- [__week01__](./week01_embeddings) __Word Embeddings__
  - Lecture: Word embeddings. Distributional semantics. Count-based (pre-neural) methods. Word2Vec: learn vectors. GloVe: count, then learn. Evaluation: intrinsic vs extrinsic. Analysis and Interpretability. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_word_emb)
  - Seminar: Playing with word and sentence embeddings
  - Homework: Embedding-based machine translation system

- **TBU** `./week03_lm` __Language Modeling__
  - Lecture: Language Modeling: what does it mean? Left-to-right framework. N-gram language models. Neural Language Models: General View, Recurrent Models, Convolutional Models. Evaluation. Practical Tips: Weight Tying. Analysis and Interpretability. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_lang_models)
  - Seminar: Build a N-gram language model from scratch
  - Homework: Neural LMs & smoothing in count-based models.
  
- **TBU** `./week03_attention` __Seq2seq and Attention__
  - Lecture: Seq2seq Basics: Encoder-Decoder framework, Training, Simple Models, Inference (e.g., beam search). Attention: general, score functions, models. Transformer: self-attention, masked self-attention, multi-head attention; model architecture. Subword Segmentation (BPE). Analysis and Interpretability: functions of attention heads; probing for linguistic structure. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_seq2seq_attn)
  - Seminar: Basic sequence to sequence model
  - Homework: Machine translation with attention
  
- **TBU** `./week04_transfer` __Transfer Learning__
  - Lecture: What is Transfer Learning? Great idea 1: From Words to Words-in-Context (CoVe, ELMo). Great idea 2: From Replacing Embeddings to Replacing Models (GPT, BERT). (A Bit of) Adaptors. Analysis and Interpretability. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_transfer)
  - Homework: fine-tuning a pre-trained BERT model


- **TBU** `./week06_llm` __Large Language Models__
  - Lecture: Scaling laws. Emergent abilities. Prompting (aka "in-context learning"): techiques that work; questioning whether model "understands" prompts. Hypotheses for why and how in-context learning works. Analysis and Interpretability.
  - Homework: manual prompt engneering and chain-of-thought reasoning

- Additional lectures to be announced!

# Contributors & course staff
Course materials and teaching performed by
- [Elena Voita](https://lena-voita.github.io) - course author
- [Mikhail Diskin] [Ignat Romanov] [Ruslan Svirschevski] - lectures
- [Valentina Broner](https://www.hse.ru/org/persons/831207784/?_gl=1%2a1hz2yht%2a_ga%2aMTg3MTM2ODIwMS4xNjk4NTEyODg5%2a_ga_D145P1R4PL%2aMTY5ODUxMjg4OC4xLjAuMTY5ODUxMjg4OC42MC4wLjA.) - course admin for on-campus students
- [Boris Kovarsky](https://github.com/kovarsky), [David Talbot](https://github.com/drt7), [Sergey Gubanov](https://github.com/esgv), [Just Heuristic](https://github.com/justheuristic) - help build course materials and/or held some classes
- [30+ volunteers](https://github.com/yandexdataschool/nlp_course/graphs/contributors) who contributed and refined the notebooks and course materials. Without their help, the course would not be what it is today
- [A mighty host of TAs](https://lk.yandexdataschool.ru/courses/2023-autumn/7.1171-avtomaticheskaia-obrabotka-tekstov/) who stoically grade hundreds of homework submissions from on-campus students each year

