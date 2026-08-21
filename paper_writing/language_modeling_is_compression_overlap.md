# Relationship to *Language Modeling Is Compression*

## Bibliographic record

Grégoire Delétang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim
Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew
Aitchison, Laurent Orseau, Marcus Hutter, and Joel Veness. “Language Modeling
Is Compression.” ICLR 2024. arXiv:2309.10668.

- ICLR paper: https://proceedings.iclr.cc/paper_files/paper/2024/file/3cbf627fa24fb6cb576e04e689b9428b-Paper-Conference.pdf
- Code: https://github.com/google-deepmind/language_modeling_is_compression

## What the paper already establishes

1. A predictive distribution can be turned into a nearly optimal lossless
   compressor with arithmetic coding; its ideal code length is the sequence
   negative log-likelihood. Conversely, code-length differences from a
   compressor can induce a predictor.
2. Pretrained foundation models can serve as offline, in-context compressors.
   The paper evaluates text, image, audio, and random bytes, and distinguishes
   raw compression rate from an adjusted rate that includes model parameters.
3. Tokenization is explicitly described as lossless “pre-compression.” The
   authors compare ASCII with BPE vocabularies from 1K to 20K on enwik9. Larger
   vocabularies shorten token sequences and pack more source information into a
   fixed token context, but enlarge the prediction alphabet; the observed net
   effect depends on model size.
4. Context length, context resets, compute, model size, and finite data volume
   are systems-level constraints on neural compression.

## Overlap and distinction

| Dimension | Delétang et al. | Our paper | Positioning consequence |
|---|---|---|---|
| Prediction and coding | Establishes the NLL/arithmetic-code correspondence and evaluates LMs as compressors. | Uses that correspondence as one layer in the system. | Do not claim that using an LLM with arithmetic coding is novel. |
| Tokenization | Treats lossless tokenization as pre-compression and varies tokenizer vocabulary on natural data. | Factorizes byte rate into tokens/source-byte and coded bits/token for a fixed pretrained tokenizer. | Claim an explicit robustness/accounting decomposition, not the idea that tokenization compresses. |
| Inputs | Natural text, image, audio, and random bytes. | Canonical, white-box adversarial token sequences plus matched text8. | The main novelty is worst-case and adversarial evaluation. |
| Objective | Reports average compression rate and studies model/tokenizer scaling. | Optimizes surprisal per decoded byte and realized serialized size. | Emphasize the mismatch between token surprisal and byte-normalized expansion. |
| Codec accounting | Separates raw rate from model-size-adjusted rate. | Separates token-table rate, model floor, finite coder loss, dictionary, and framing; model weights are shared and excluded. | State the amortization assumption and avoid calling the reported rate fully adjusted. |
| Validity | Encodes source data through a tokenizer. | Generates token IDs adversarially and therefore enforces prefix canonicality, `T(D(x)) = x`, plus serialized round trips. | Canonical adversarial construction is a distinct methodological contribution. |
| Robustness | Notes by injectivity that not all sequences can compress, and uses random data as a baseline. | Searches for high-cost inputs, studies masking, and proposes guarded fallback. | Frame the work as robustness of the composition, not average-case compressor benchmarking. |

## Claims to avoid

- “We are the first to view tokenization as compression.”
- “We are the first to use an LLM with arithmetic coding.”
- “We establish that language modeling and compression are equivalent.”
- “Perplexity directly determines source-byte compression.” The equivalence is
  at the modeled-symbol level; variable decoded token lengths and all transmitted
  metadata matter for the end-to-end byte rate.
- “Our final file rate includes the whole system.” It currently excludes shared
  model weights and should be called a raw offline or amortized rate.

## Defensible novelty statement

> Prior work establishes language-model arithmetic coding and treats
> tokenization as lossless pre-compression. We study a different question: the
> adversarial robustness of their composition. We derive and optimize
> source-byte-normalized objectives, enforce canonical token round trips, and
> attribute realized expansion across the token table, predictor, arithmetic
> coder, dictionary, and container.

## Experimental implications

- Keep both bits/token and bits/source-byte. Delétang et al. makes the former
  meaningful as log loss; our contribution is showing why it is insufficient
  when token byte lengths vary.
- Continue reporting a one-byte alphabet experiment: it removes tokenization
  gain and cleanly exposes predictive expansion.
- Explicitly label model weights as excluded/shared. A future amortization plot
  could add model bytes divided by corpus bytes, paralleling the adjusted rate in
  Delétang et al.
- Treat context resets as part of the codec configuration. Their evaluation
  shows that reset/chunk policy changes compression, so matched context policy is
  necessary for natural/adversarial comparisons.
- A tokenizer/model matrix would complement, rather than duplicate, their
  tokenizer ablation: they vary tokenizer design on natural inputs; we would
  measure adversarial robustness across fixed pretrained pairs.

## Reference shortlist

The four-page manuscript should prioritize the five references now cited:

1. Delétang et al. (ICLR 2024) for prediction–compression equivalence,
   foundation-model compression, tokenizer pre-compression, and adjusted rates.
2. Schmidt et al. (2026 manuscript) for global queryable LLM token tables.
3. Shannon (1948) for the source-coding foundation.
4. Witten, Neal, and Cleary (1987) for practical arithmetic coding.
5. Valmeekam et al. (2023) for closely related LLaMA-based lossless text
   compression.

If references are excluded from the workshop’s page limit, useful additions are
FineZip (Mittu et al., 2024) for practical throughput limitations; Rissanen
(1976) for arithmetic coding; Sennrich et al. (2016) for BPE; and Bellard (2021)
or TRACE (Mao et al., 2022) for neural/Transformer compressors.
