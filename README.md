# Offensive language and hate speech in Arabic tweets

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)

We fine-tuned MARBERT, a BERT model pretrained on Arabic tweets, to catch two things in Arabic Twitter text: offensive language, and the harder case of hate speech. The data comes from the OSACT 2022 shared task on fine-grained hate speech detection.

This was a team project with four other people, and this repo is a fork of [the original](https://github.com/mahmoudramadan0040/TextCalssificationUsingBERT) by Mahmoud Ramadan. I'm one of the five team members.

## The problem

Moderating online abuse at scale needs automatic classifiers. Arabic makes this harder than English because tweets mix dialects (Egyptian, Gulf, Levantine and more) and the spelling isn't standardised.

The dataset has about 13,000 tweets, each judged by three annotators. Around 35% are offensive, and only 11% are hate speech. That imbalance is why hate speech is the harder subtask.

| Subtask | Question | Labels |
|---|---|---|
| A | Is the tweet offensive? | OFF / NOT_OFF |
| B | Does it contain hate speech? | HS / NOT_HS |

## What we did

1. **Cleaned the text.** We removed emoji and the label tags that were left in the tweets, then converted the labels to 0 and 1 (`PHASE1(preparing&preprocessing).ipynb`).
2. **Preprocessed with the AraBERT pipeline** for MARBERT's tokenizer.
3. **Fine-tuned MARBERT** with a classification head, one model per subtask.
4. **Evaluated** on 10% of the data held out with a fixed random seed (42).

Training settings for both models: learning rate 5e-5, batch size 32 with gradient accumulation of 5, 4 epochs, max sequence length 266, mixed precision (fp16) on a Colab GPU. We did not tune hyperparameters.

## Results

Scores on the held-out 10%, after the final epoch:

| Subtask | Macro F1 | Accuracy |
|---|---|---|
| A: Offensive language | **0.880** | 0.880 |
| B: Hate speech | **0.928** | 0.928 |

Hate speech scored higher than offensive language, even though it's the rarer class. For subtask B the best epoch (the third) reached a macro F1 of 0.934. The final epoch is 0.928. The tables below show every epoch.

<details>
<summary>Per-epoch scores</summary>

**Subtask A (offensive)**

| Epoch | Validation loss | Macro F1 | Accuracy |
|---|---|---|---|
| 1 | 0.332 | 0.866 | 0.866 |
| 2 | 0.330 | 0.879 | 0.879 |
| 3 | 0.433 | 0.877 | 0.877 |
| 4 | 0.509 | 0.880 | 0.880 |

**Subtask B (hate speech)**

| Epoch | Validation loss | Macro F1 | Accuracy |
|---|---|---|---|
| 1 | 0.351 | 0.859 | 0.860 |
| 2 | 0.278 | 0.909 | 0.909 |
| 3 | 0.223 | 0.934 | 0.934 |
| 4 | 0.252 | 0.928 | 0.928 |

</details>

## Caveats

- **These are not leaderboard scores.** The numbers come from our own 90/10 split of the shared-task data, not from the official OSACT test set, so they can't be compared directly with other teams' results.
- **We used the held-out split for monitoring and for reporting.** There's no separate test set, so the scores are slightly optimistic.
- **Validation loss rises in subtask A.** It goes from 0.33 in epoch 2 to 0.51 in epoch 4 while F1 stays flat. That's a sign the model is overfitting, and fewer epochs or early stopping would be better.
- **Single run.** We trained once with one seed and report no confidence intervals.

## How to run

The notebooks were written for Google Colab with a GPU.

1. Download the training and dev data from the [OSACT 2022 task page](https://alt.qcri.org/resources/OSACT2022/) or use the CSV files in this repo.
2. Open `OFF_NOT_OFF_Model.ipynb` (subtask A) or `HS_NOT_HS_Model.ipynb` (subtask B) in Colab.
3. Upload the matching CSV, and change the path in the `pd.read_csv` cell if needed.
4. Run the cells in order. The first cell installs `transformers==4.2.1`, `optuna`, `farasapy`, `pyarabic` and `arabert`, and downloads MARBERT from [Hugging Face](https://huggingface.co/UBC-NLP/MARBERT).

Training takes about 12 minutes for subtask A and 5 minutes for subtask B on a Colab GPU.

## What I'd like to do next

Look inside the model: which words push a tweet towards "offensive", whether those explanations hold up on different dialects, and whether they stay stable when the text is noisy.

## Credits

- The shared task data is from [OSACT 2022](https://arxiv.org/pdf/2201.06723.pdf).
- The model is [MARBERT](https://huggingface.co/UBC-NLP/MARBERT), from the paper *ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic* (ACL 2021).
- The tweets include offensive language, so be aware of that when you open the data.
