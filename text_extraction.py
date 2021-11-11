"""
Title: Text Extraction with BERT
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/23
Last modified: 2020/05/23
Description: Fine tune pretrained BERT from HuggingFace Transformers on SQuAD.
"""
"""
## Introduction

This demonstration uses SQuAD (Stanford Question-Answering Dataset).
In SQuAD, an input consists of a question, and a paragraph for context.
The goal is to find the span of text in the paragraph that answers the question.
We evaluate our performance on this data with the "Exact Match" metric,
which measures the percentage of predictions that exactly match any one of the
ground-truth answers.

We fine-tune a BERT model to perform this task as follows:

1. Feed the context and the question as inputs to BERT.
2. Take two vectors S and T with dimensions equal to that of
   hidden states in BERT.
3. Compute the probability of each token being the start and end of
   the answer span. The probability of a token being the start of
   the answer is given by a dot product between S and the representation
   of the token in the last layer of BERT, followed by a softmax over all tokens.
   The probability of a token being the end of the answer is computed
   similarly with the vector T.
4. Fine-tune BERT and learn S and T along the way.

**References:**

- [BERT](https://arxiv.org/pdf/1810.04805.pdf)
- [SQuAD](https://arxiv.org/abs/1606.05250)
"""
"""
## Setup
"""
import os
import re
import string

from metaflow import FlowSpec, step, conda

max_len = 384

class SquadExample:
    """
    ## Preprocess the data

    1. Go through the JSON file and store every record as a `SquadExample` object.
    2. Go through each `SquadExample` and create `x_train, y_train, x_eval, y_eval`.
    """
    def __init__(self, question, context, start_char_idx, answer_text, all_answers, tokenizer):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False
        self.tokenizer = tokenizer

    def preprocess(self):
        context = self.context
        question = self.question
        answer_text = self.answer_text
        start_char_idx = self.start_char_idx
        tokenizer = self.tokenizer

        # Clean context, answer and question
        context = " ".join(str(context).split())
        question = " ".join(str(question).split())
        answer = " ".join(str(answer_text).split())

        # Find end character index of answer in context
        end_char_idx = start_char_idx + len(answer)
        if end_char_idx >= len(context):
            self.skip = True
            return

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1

        # Tokenize context
        tokenized_context = tokenizer.encode(context)

        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)

        if len(ans_token_idx) == 0:
            self.skip = True
            return

        # Find start and end token index for tokens from answer
        start_token_idx = ans_token_idx[0]
        end_token_idx = ans_token_idx[-1]

        # Tokenize question
        tokenized_question = tokenizer.encode(question)

        # Create inputs
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = max_len - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.context_token_to_char = tokenized_context.offsets


def create_squad_examples(raw_data, tokenizer):
    squad_examples = []
    for item in raw_data["data"]:
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                answer_text = qa["answers"][0]["text"]
                all_answers = [_["text"] for _ in qa["answers"]]
                start_char_idx = qa["answers"][0]["answer_start"]
                squad_eg = SquadExample(
                    question, context, start_char_idx, answer_text, all_answers, tokenizer
                )
                squad_eg.preprocess()
                squad_examples.append(squad_eg)
    return squad_examples


def create_inputs_targets(squad_examples):
    import numpy as np

    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y


"""
Create the Question-Answering Model using BERT and Functional API
"""
def create_model():
    ## BERT encoder
    from transformers import TFBertModel
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy

    encoder = TFBertModel.from_pretrained("bert-base-uncased")

    ## QA Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]

    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
    start_logits = layers.Flatten()(start_logits)

    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
    end_logits = layers.Flatten()(end_logits)

    start_probs = layers.Activation(keras.activations.softmax)(start_logits)
    end_probs = layers.Activation(keras.activations.softmax)(end_logits)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=5e-5)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model


def normalize_text(text):
    """
    ## Create evaluation Callback

    This callback will compute the exact match score using the validation data
    after every epoch.
    """
    text = text.lower()

    # Remove punctuations
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)

    # Remove articles
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)

    # Remove extra white space
    text = " ".join(text.split())
    return text


class TextExtractionBERT(FlowSpec):

    @conda(libraries={"tensorflow": "2.6.0"})
    @step
    def start(self):
        """
        ## Load the data
        """
        from tensorflow import keras
        import json

        train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
        train_path = keras.utils.get_file("train.json", train_data_url)
        eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
        eval_path = keras.utils.get_file("eval.json", eval_data_url)

        with open(train_path) as f:
            self.raw_train_data = json.load(f)

        with open(eval_path) as f:
            self.raw_eval_data = json.load(f)

        self.next(self.setup_tokenizer)

    @conda(libraries={"tensorflow": "2.6.0", "transformers": "4.11.3", "tokenizers": "0.10.3"})
    @step
    def setup_tokenizer(self):
        """
        ## Set-up BERT tokenizer
        """
        from tokenizers import BertWordPieceTokenizer
        from transformers import BertTokenizer
        # Save the slow pretrained tokenizer
        self.slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        save_path = "bert_base_uncased/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.slow_tokenizer.save_pretrained(save_path)

        # Load the fast tokenizer from saved file
        self.tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)

        self.next(self.create_train_points, self.create_eval_points)


    @conda(libraries={"tokenizers": "0.10.3", "numpy": "1.21.4"})
    @step
    def create_train_points(self):
        import numpy as np

        train_squad_examples = create_squad_examples(self.raw_train_data, self.tokenizer)
        self.x_train, self.y_train = create_inputs_targets(train_squad_examples)
        print(f"{len(train_squad_examples)} training points created.")
        self.next(self.join)


    @conda(libraries={"tokenizers": "0.10.3", "numpy": "1.21.4"})
    @step
    def create_eval_points(self):
        import numpy as np

        self.eval_squad_examples = create_squad_examples(self.raw_eval_data, self.tokenizer)
        self.x_eval, self.y_eval = create_inputs_targets(self.eval_squad_examples)
        print(f"{len(self.eval_squad_examples)} evaluation points created.")
        self.next(self.join)

    @conda(libraries={"numpy": "1.21.4"})
    @step
    def join(self, inputs):
        import numpy

        self.x_train, self.y_train = inputs.create_train_points.x_train, inputs.create_train_points.y_train
        self.x_eval, self.y_eval = inputs.create_eval_points.x_eval, inputs.create_eval_points.y_eval

        self.next(self.end)


    @conda(libraries={"tensorflow": "2.6.0", "numpy": "1.21.4"})
    @step
    def end(self):
        """
        ## Train and Evaluate
        """
        import numpy as np
        from tensorflow import keras


        class ExactMatch(keras.callbacks.Callback):
            """
            Each `SquadExample` object contains the character level offsets for each token
            in its input paragraph. We use them to get back the span of text corresponding
            to the tokens between our predicted start and end tokens.
            All the ground-truth answers are also present in each `SquadExample` object.
            We calculate the percentage of data points where the span of text obtained
            from model predictions matches one of the ground-truth answers.
            """

            def __init__(self, x_eval, y_eval, eval_squad_examples):
                self.x_eval = x_eval
                self.y_eval = y_eval
                self.eval_squad_examples = eval_squad_examples

            def on_epoch_end(self, epoch, logs=None):
                eval_squad_examples = self.eval_squad_examples
                pred_start, pred_end = self.model.predict(self.x_eval)
                count = 0
                eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip == False]
                for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
                    squad_eg = eval_examples_no_skip[idx]
                    offsets = squad_eg.context_token_to_char
                    start = np.argmax(start)
                    end = np.argmax(end)
                    if start >= len(offsets):
                        continue
                    pred_char_start = offsets[start][0]
                    if end < len(offsets):
                        pred_char_end = offsets[end][1]
                        pred_ans = squad_eg.context[pred_char_start:pred_char_end]
                    else:
                        pred_ans = squad_eg.context[pred_char_start:]

                    normalized_pred_ans = normalize_text(pred_ans)
                    normalized_true_ans = [normalize_text(_) for _ in squad_eg.all_answers]
                    if normalized_pred_ans in normalized_true_ans:
                        count += 1
                acc = count / len(self.y_eval[0])
                print(f"\nepoch={epoch + 1}, exact match score={acc:.2f}")


        exact_match_callback = ExactMatch(self.x_eval, self.y_eval, self.eval_squad_examples)

        model = create_model()
        model.summary()

        model.fit(
            self.x_train,
            self.y_train,
            epochs=1,  # For demonstration, 3 epochs are recommended
            verbose=2,
            batch_size=64,
            callbacks=[exact_match_callback],
        )


if __name__ == "__main__":
    TextExtractionBERT()
