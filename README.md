# YSDA Natural Language Processing course
* This is the 2021 version. For previous year' course materials, go to [this branch](https://github.com/yandexdataschool/nlp_course/tree/2020)
* Lecture and seminar materials for each week are in ./week* folders, see README.md for materials and instructions
* YSDA homework deadlines will be listed in Anytask ([read more](https://github.com/yandexdataschool/nlp_course/wiki/Homeworks-and-grading)).
* Any technical issues, ideas, bugs in course materials, contribution ideas - add an [issue](https://github.com/yandexdataschool/nlp_course/issues)
* Installing libraries and troubleshooting: [this thread](https://github.com/yandexdataschool/nlp_course/issues/1).


# [NLP Course For You](https://lena-voita.github.io/nlp_course.html)


# Syllabus
- [__week01__](./week01_embeddings) __Word Embeddings__
  - Lecture: Word embeddings. Distributional semantics. Count-based (pre-neural) methods. Word2Vec: learn vectors. GloVe: count, then learn. Evaluation: intrinsic vs extrinsic. Analysis and Interpretability. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_word_emb)
  - Seminar: Playing with word and sentence embeddings
  - Homework: Embedding-based machine translation system

- [__week02__](./week02_classification) __Text Classification__
  - Lecture: Text classification: introduction and datasets. General framework: feature extractor + classifier. Classical approaches: Naive Bayes, MaxEnt (Logistic Regression), SVM. Neural Networks: General View, Convolutional Models, Recurrent Models. Practical Tips: Data Augmentation. Analysis and Interpretability. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_text_clf)
  - Seminar: Text classification with convolutional NNs.
  - Homework: Statistical & neural text classification.
  
- [__week03__](./week03_lm) __Language Modeling__
  - Lecture: Language Modeling: what does it mean? Left-to-right framework. N-gram language models. Neural Language Models: General View, Recurrent Models, Convolutional Models. Evaluation. Practical Tips: Weight Tying. Analysis and Interpretability. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_lang_models)
  - Seminar: Build a N-gram language model from scratch
  - Homework: Neural LMs & smoothing in count-based models.
  
- [__week04__](./week04_seq2seq) __Seq2seq and Attention__
  - Lecture: Seq2seq Basics: Encoder-Decoder framework, Training, Simple Models, Inference (e.g., beam search). Attention: general, score functions, models. Transformer: self-attention, masked self-attention, multi-head attention; model architecture. Subword Segmentation (BPE). Analysis and Interpretability: functions of attention heads; probing for linguistic structure. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_seq2seq_attn)
  - Seminar: Basic sequence to sequence model
  - Homework: Machine translation with attention
  
- [__week05__](./week05_transfer) __Transfer Learning__
  - Lecture: What is Transfer Learning? Great idea 1: From Words to Words-in-Context (CoVe, ELMo). Great idea 2: From Replacing Embeddings to Replacing Models (GPT, BERT). (A Bit of) Adaptors. Analysis and Interpretability. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_transfer)


- [__week06__](./week06_da) __Domain Adaptation__
  - Lecture: General theory. Instance weighting. Proxy-labels methods. Feature matching methods. Distillation-like methods.
  - Seminar+Homework: BERT-based NER domain adaptation
  
- [__week07__](./week07_compression) __Model compression and acceleration__

- [__week08__](./week08_em) __Probabilistic inference, generative models and hidden variables__

- [__week09__](./week09_mt) __Machine translation__

- [__week10__](./week10_relation_extraction) __Relation extraction__

- [__week11__](./week11_summarization) __Summarization__

- [__week12__](./week12_style) __Style Transfer__

- [__week13__](./week13_conversation) __Dialogue systems__

- [__week14__](./week14_ai_ml_generated_art) __AI & ML generated art__


# Contributors & course staff
Course materials and teaching performed by
- [Elena Voita](https://lena-voita.github.io) - course admin, lectures, seminars, homeworks
- [Boris Kovarsky](https://github.com/kovarsky) - lectures, seminars, homeworks
- [David Talbot](https://github.com/drt7) - lectures, seminars, homeworks
- [Just Heuristic](https://github.com/justheuristic) - lectures, seminars, homeworks
- [Alexey Tikhonov @altsoph](https://github.com/altsoph)
- [Michael Sejr Schlichtkrull](http://michschli.github.io)
- [Arthur Bražinskas](https://github.com/abrazinskas/)
- [Ivan Yamshchikov](https://twitter.com/kr0niker)
- [Nikolay Zinov](https://github.com/nzinov)

# Authors and contributors of previous years
- [Sergey Gubanov](https://github.com/esgv)
- [Vyacheslav Alipov](https://research.yandex.com/people/301832)
- Vladimir Kirichenko
- Andrey Zhigunov
- Pavel Bogomolov
