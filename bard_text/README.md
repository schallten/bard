# bard_text

A tiny language model trained from scratch in Go, on *Hamlet*. Standard
library only — no ML frameworks, no dependencies.

The model looks at **two previous words** and predicts the next one. That makes
it a soft trigram: for each word pair it learns which word tends to follow. It
is deliberately dumb, and the point is to understand every line of it.

## How it works

The whole pipeline lives in `main.go` and runs in order:

1. **Load and clean tokens** (`clean.go`, `tokenizer.go`) — lowercase, fix
   punctuation, split punctuation into its own tokens, build the vocabulary.
2. **Encode** — turn the token stream into integer IDs.
3. **Build examples** (`dataset.go`) — slide a 3-wide window over the IDs, so
   every example is `(word, word) -> next word`.
4. **Forward pass** (`model.go`) — score every candidate word, turn scores into
   probabilities with softmax.
5. **Loss** (`loss.go`) — cross-entropy: how little probability the model gave
   the correct word.
6. **Train** (`train.go`) — for each example, the correction for every candidate
   is `guessed probability − 1 (if it was the correct word)`. Nudge the bias and
   the two context rows by that amount, scaled by the learning rate. Repeat.
7. **Sample** (`gen.go`) — walk the model from a two-word start, drawing the next
   word from its probabilities (with temperature).

## Run it

```
cd bard_text
go run .
```

This regenerates `hamlet_clean_tokens.txt`, trains for 10 epochs, runs a
gradient check, and then drops into an interactive prompt. Takes about 25
seconds before the prompt appears.

At the prompt, type a phrase and press enter. The model only uses the **last two
tokens** as context, so `to be or not to be` continues from `to be`. A single
word is paired with `.` so there are always two. Words that aren't in Hamlet's
vocabulary are reported instead of guessed at. Output is capped at 40 words, and
an empty line (or Ctrl-D) quits.

```
> the tragedy
the tragedy , my father ? hamlet . now the we'll to stick , my lord ...
> zzzz
"zzzz" is not in the vocabulary
```

```
go vet ./...
```

## Files

| File | Purpose |
|------|---------|
| `main.go` | Wires the pipeline together |
| `clean.go` | Normalizes and re-tokenizes the corpus |
| `tokenizer.go` | Vocabulary, encode/decode, file loading |
| `dataset.go` | Sliding-window `(context, target)` examples |
| `model.go` | Weights, logits, softmax, forward pass |
| `loss.go` | Cross-entropy |
| `train.go` | SGD step, training loop, gradient check |
| `gen.go` | Temperature sampling and text generation |
| `cli.go` | Interactive prompt: phrase in, text out |
| `decision.txt` | Cleaning and tokenization decisions |
| `notes.md` | Long-form explanation, worked examples, results |

## Data

| File | What |
|------|------|
| `hamlet.txt` | Original Project Gutenberg text (untouched) |
| `hamlet_tokens.txt` | v1 tokens: whitespace split, case preserved, boilerplate and stage directions removed |
| `hamlet_clean_tokens.txt` | v2 tokens: lowercased, punctuation split out — **regenerated on every run** |

Source: https://www.gutenberg.org/cache/epub/1524/pg1524.txt

## Results

Tokenizer v2, `lr = 0.1`, 10 epochs, last 10% of the play held out for
validation (35,384 train / 3,930 val examples):

| | train | val |
|---|---|---|
| Loss before | 8.4667 | 8.4669 |
| Loss after | 4.0930 | 5.3464 |
| Perplexity | ~60 | ~210 |

A uniform guess over the vocabulary scores `ln(4755) ≈ 8.47` and a perplexity of
`4755`. Training loss keeps falling, but validation loss bottoms out around
**5.34** at epoch 8 and then creeps back up — the model starts to memorize.
That gap is the honest number: it generalizes to unseen word pairs only up to a
point.

Sample at temperature 0.5:

```
the tragedy , sir , - who ? hamlet . hamlet . what is a - will i do not that i
have seen . - now , my lord . hamlet . hamlet . then is the matter , my lord
```

## What it can't do

The weights are additive and only two words wide, so it memorizes local
statistics and cannot, for example, remember what a sentence is about. It will
loop (`i am i am`). That ceiling is the honest lesson of the project: making it
better needs more than tuning — it needs context that composes, and weights that
generalize across similar words. That is a different model, not a bigger table.

## Requirements

Go 1.27.1 (per `go.mod`). No third-party packages.
