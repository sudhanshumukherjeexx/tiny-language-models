# Where TinyLM fails

Every example below is a **real, unedited output**. Part A covers the TinyLM-27M model trained
with the current pipeline, on the fixed prompt suite (`evaluation/prompts.json`, v1;
outputs in `results/runs/tiny-27m/evaluation/generation_samples.jsonl`; prompt id, preset and
seed in brackets). Part B covers the legacy notebook checkpoint
(`results/runs/legacy-colab-a100/generation_samples.jsonl`; sample index in brackets).

Greedy outputs (`preset=greedy`) reproduce exactly with `scripts/generate.py --preset greedy`.
Where a failure is not consistent across runs, the count is given: each prompt has 1 greedy and
3 sampled (`balanced`: T 0.8, top-p 0.9) generations. "Likely cause" entries are hypotheses
consistent with the model, data and decoding. None was tested with an intervention.

# Part A: TinyLM-27M (test loss 1.384)

### A1. Contradicting the prompt  `[sens-01b greedy]`, 1 of 4 runs

> **Prompt:** There once was a little boy called Leo, and he **really loved trains.**
> … "But **I don't like trains**. They are too loud and too fast," Leo said.

- **Observed failure:** a trait stated in the prompt is reversed within ~120 tokens.
- **Likely cause:** "Leo was scared → I don't like X" is a frequent TinyStories pattern, and once
  the model has generated "scared", local context outweighs the opening sentence.
- **Note:** the three sampled continuations of the same prompt keep Leo's love of trains, and
  the near-paraphrase `sens-01a` produces a consistent story. Prompt sensitivity is real but not
  systematic here.

### A2. Ignoring an instruction one sentence later  `[open-02 greedy]`

> Lily wanted to play on the swing, but her mom said, "No, Lily. You can't play on the swing."
> Lily was sad, but **she listened to her mom. She played on the swing** and had fun.

- **Observed failure:** "listened to her mom" is immediately followed by doing what mom forbade.
- **Likely cause:** "listened to her mom" and "played on the swing and had fun" are both
  high-probability continuations of their local context; the model does not track the
  prohibition as state.

### A3. Breaking a physical constraint from the prompt  `[char-01]`, 2 of 4 runs

> **Prompt:** Max was a brown dog who **could not swim**. … with his friend Bella, a white duck.
> greedy: "He jumped into the water… Max screamed and **tried to swim back** to the shore."
> seed 2: "Max said to Bella, 'I want to swim too!'… **They swam to the boat**…"

- **Observed failure:** the greedy run half-respects the constraint (Max is in trouble in the
  water). Seed 2 ignores it entirely.
- **Possible mitigation:** none at the decoding level. Constraint tracking over ~100 tokens is a
  capacity limit at 27M parameters.

### A4. Degenerate repetition and self-contradiction in dialogue  `[char-01 greedy]`, `[end-02 greedy]`

> Bella said, "No, Max, that's not a fish. That's a fish. It's a fish. It's not a fish. It's a fish."

> He did not see the big wave coming. He did not see the big wave that was coming.

- **Observed failure:** greedy decoding loops on a short phrase, contradicting itself in the
  process. Greedy generations have 3.3 % repeated 4-grams vs. 1.6 % for sampling
  (`lexical_metrics.json`).
- **Likely cause:** the likelihood trap: under greedy decoding, repeating a just-generated phrase
  is often the single most probable continuation.
- **Possible mitigation:** sampling (`balanced` preset) or a mild repetition penalty.

### A5. Physically impossible outcomes  `[cause-02 greedy]`, `[open-01 greedy]`

> Sam dropped the glass on the floor and it **broke**. … Mom took the glass and put it back
> together. … soon **the glass was fixed**.

> she saw a big, red ball **in the sky**. … They went to the store and **bought the ball**. … the
> ball started to get too high up in the sky.

- **Observed failure:** the first clause is correct ("it broke"), and then the story follows the
  much more frequent "we can fix it" script. In the second, a ball floats in the sky, gets
  bought at a store, and floats away again.
- **Likely cause:** script completion without a world model. "Broken toy → mom fixes it" is
  frequent in TinyStories.

### A6. Long-context collapse and role confusion  `[long-01]`, 4 of 4 runs

> **Prompt (≈200 tokens):** Rosie… had a pet turtle called Shelly… Shelly was not in her box…
> Rosie saw a seagull standing very still on a rock… Rosie ran to the rock and
> greedy (33 tokens, then end of story): "saw a big, yellow shell. **Shelly** was so happy! **She
> picked it up and showed it to her grandpa.**"
> seed 0 (39 tokens): "…She had a good day with her **yellow turtle**."

- **Observed failure:** every run ends the story within 33–51 tokens (`eos_rate` 1.0, mean 34
  words, vs. 124–132 words overall). The lost turtle becomes the protagonist who picks up a
  shell, the seagull setup is dropped, and "yellow" migrates from the house to the turtle.
- **Likely cause:** prompts this long are rare openings in training (stories average 220
  tokens). The model behaves as if it is near the end of a story and closes it.
- **Possible mitigation:** training on longer stories, or evaluating with shorter setups. This is
  the clearest capability limit in the suite.

### A7. Rare words disintegrate into subword fragments  `[rare-02 greedy]`

> The archaeologist carefully brushed the dust off the ancient pottery and put it in a special
> box. … The **archae arched** around the room… The **archaeist** was so happy… thanked the **archae**

- **Observed failure:** the model cannot reproduce a word it has rarely seen, so it emits
  plausible fragments of the BPE pieces.
- **Likely cause:** "archaeologist" is split into several rare subword tokens with weak
  embeddings. Byte-level BPE guarantees encodability, not competence.
- **Note:** `rare-01` (astronaut) keeps the vocabulary but drifts into fairy-tale register (the
  astronaut talks to the stars).

### A8. Out-of-domain prompts: no knowledge, story register  `[ood-01/02/03]`, 12 of 12 runs

> "The quarterly revenue of the company increased because **he was the best singer in the world.**"
> "To compute the derivative of x squared, we**andered the room**." (then a story about an x-ray)
> "Question: What is the capital of France? Answer: **Neverse! Neverse France is a special kind of capture.**"
> seed 0: "Enough: Oh no! Oh no! Enough: Oh no! Oh no, Yeah! Yay! Yay!…"

- **Observed failure:** every out-of-domain prompt is turned into a children's story, sometimes
  after a non-word ("andered") or a degenerate stretch.
- **Likely cause:** expected. The model has seen no text outside TinyStories and was never
  trained to follow instructions. This row exists to document that the model is not an assistant.

### A9. Robust behaviour (included so the picture is not one-sided)

- **Trailing punctuation:** "Once upon a time" and "Once upon a time," give the *same* greedy
  story (`sens-02a/b`).
- **Simple causality is often right:** "Sam dropped the glass… and it **broke**"; "It started to
  rain, so Mia… splashed in the puddles"; a lost teddy bear is found and the story resolves
  (`end-01`).
- **Story structure:** 74 % of sampled generations end with an end-of-story token within 200 new
  tokens, usually after a resolution.

# Part B: legacy notebook checkpoint

No examples were selected for being bad: the notebook printed 14 samples in total, and every
failure found in them is listed here. (Seven of the 14 end mid-sentence because they hit
`max_new_tokens`; that is truncation, not a model failure.)

### 1. Repetition loops under beam search  `[7]`, beam search (4 beams)

> After a while, Lily's mommy said, **"Lily, it's time to go home now." Lily didn't want to leave yet, but she knew she had to listen to her mommy and daddy.**
> When they got home, Lily's mommy said, **"Lily, it's time to go to bed now." Lily didn't want to go to bed yet, but she knew she had to listen to her mommy and daddy.**

- **Observed failure:** the same sentence template repeats with one slot changed.
- **Likely cause:** beam search maximizes sequence likelihood, and for a small model on formulaic data the highest-likelihood continuation is often a template the model has just used. This is the "likelihood trap" known from larger models too.
- **Possible mitigation:** sampling (top-p) instead of beam search, a repetition penalty or `no_repeat_ngram_size`. The `sentence_repetition_ratio` metric flags this pattern automatically.

### 2. Near-verbatim repetition under sampling  `[8]`, T=0.8, top-p 0.9, top-k 50

> Lily didn't want to take a bath because she was having so much fun. … Lily didn't want to take a bath because she was having too much fun.

- **Observed failure:** a sentence is restated two sentences later.
- **Likely cause:** with weak long-range planning, re-attending to an earlier sentence is a high-probability move. The TinyStories training text itself contains deliberate repetition.
- **Possible mitigation:** a mild repetition penalty (the `[6]` sample with 1.15 shows no exact repeats, though one sample proves nothing), or more training tokens.

### 3. Character and goal inconsistency  `[11]`, T=0.8, top-p 0.9, top-k 50

> The big dog got angry and **chased the little girl** … The big dog came over and licked her face. The little girl was happy that **the big dog saved her toy.** … The dog ran away and **the man took his toy.**

- **Observed failure:** the antagonist turns into a rescuer with no motivation, and the toy's owner changes between sentences.
- **Likely cause:** each sentence is locally plausible, but the model keeps no stable representation of who wants what. 27M parameters and next-token training give limited state tracking.
- **Possible mitigation:** more capacity or training tokens. The `char-*` prompts in the suite measure this directly.

### 4. Contradictory physical statements  `[0]` and `[13]`

> She … saw a big, round ball. **It was so big that she could barely fit in it.** `[0]`

> He heard a loud noise **coming from below him**. It was a bird who was stuck **on the roof** of a nearby house. … He **hopped over to the roof** `[13]`

- **Observed failure:** spatial and physical relations contradict each other (fitting inside a ball, a noise from below that comes from a roof, a rabbit hopping onto a roof).
- **Likely cause:** high-frequency collocations ("so big that she could barely…", "noise coming from…") get completed without a world model to check them against.
- **Possible mitigation:** not fixable by decoding settings. This is a capacity and data limitation.

### 5. Broken causal reasoning  `[6]` and `[10]`

> her mom said no because **it was too expensive for Lily's teeth.** `[6]`

> The little cat was very hungry, so she went to the kitchen and opened the door. **But she was too far away to see the cat.** `[10]`

- **Observed failure:** "because" and "but" link clauses that have no causal relationship. In `[10]` the cat cannot see itself.
- **Likely cause:** the model has learned the *form* of causal connectives ("no because it was too …") and fills the slot with a frequent adjective from a different script ("bad for your teeth").
- **Possible mitigation:** the `cause-*` prompts quantify this. Larger models trained on TinyStories are reported to improve here (Eldan & Li, 2023), but this repo does not test that.

### 6. Moral that contradicts the story  `[13]`

> Benny felt proud that he could help the bird. … **And that's the moral of the story: it's always good to ask for help when you need it.**

- **Observed failure:** Benny *gave* help and never asked for it, yet the moral is about asking for help.
- **Likely cause:** "moral of the story" endings are frequent in TinyStories, and the model reproduces a common moral rather than one derived from its own plot.
- **Possible mitigation:** none at the decoding level.

### 7. Incoherent dialogue and pronoun confusion  `[12]`

> She heard her mom come home and saw **her** with the paint all over the floor. **She** was angry and sad. "Lily, what did you do?" her mom asked. "… **Why did you ask me?**"

- **Observed failure:** ambiguous pronouns, and a question that contradicts the situation (Lily never asked).
- **Likely cause:** with two female characters, "she/her" resolution is hard. The question template "Why did you …?" gets completed with a frequent verb.

### 8. Grammar slips and non-words (more frequent at lower top-k / higher temperature)

> She made an **adjustic** plan. `[4]`, top-k 40

> "**I'm bigger than you to get the ice cream!**" … "**You're just a boy!**" (said by Max, a boy, to Lily, a girl) `[5]`, T=1.3

> found some candy that **were** just right `[6]`

- **Observed failure:** an invented word, an ungrammatical comparative, a misgendered taunt, and an agreement error.
- **Likely cause:** byte-level BPE can compose any string from subword pieces, so sampling can produce plausible-looking non-words. High temperature flattens the distribution toward these low-probability continuations.
- **Possible mitigation:** lower temperature or top-p sampling. The `low_temperature`/`balanced` presets trade diversity for these errors (compare Distinct-n against the judge or human grammar scores).

### 9. Dropped narrative threads  `[9]`

> They did not see **the big hole in the ground.** They only saw the shiny rock. … (the hole is never mentioned again before the sample ends)

- **Observed failure:** a setup is introduced and abandoned. The sample was cut off by the 170-token limit, so this is suggestive, not conclusive.
- **Possible mitigation:** evaluate with longer `max_new_tokens`. The `long-01` prompt tests whether details survive ~200 tokens of context.

### 10. Truncation is not a model failure

Seven of the 14 samples end mid-sentence (e.g. `[0]` "…Can we come back to the park tomorrow?" Her"). They hit `max_new_tokens` before the model emitted an end-of-story token. The evaluation pipeline records `finished_with_eos` for every generation, and lexical summaries report the `eos_rate`, so truncation and premature endings are measured, not guessed.
