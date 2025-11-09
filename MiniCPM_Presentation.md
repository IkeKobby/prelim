# MiniCPM: Unveiling the Potential of Small Language Models
## With Scalable Training Strategies

**Presenter:** Isaac Kobby Anni  
**Date:** January 2025  
**Department:** Computer Science & Technology

---

## Slide 1: Title Slide

# MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies

**Authors:** Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, et al.  
**Affiliation:** Tsinghua University & Modelbest Inc.

**Key Insight:** Small models can compete with models 3-5× larger through optimal training strategies

---

## Slide 2: The Problem Statement

### Current Landscape

**LLM Training Challenges:**
- 💰 **Expensive**: Trillion-parameter models cost millions to train
- 🚫 **Limited Access**: Only well-funded organizations can experiment
- 🖥️ **Deployment Issues**: Too large for edge devices and personal computers

**Key Question:**
> Can small language models achieve comparable performance to large models through better training strategies?

---

## Slide 3: MiniCPM Overview

### What is MiniCPM?

**Two Model Variants:**
- **MiniCPM-1.2B**: 1.2 billion non-embedding parameters
- **MiniCPM-2.4B**: 2.4 billion non-embedding parameters

**Core Achievement:**
- Ranks #1 in their respective size categories (1B and 2B)
- **Performance comparable to 7B-13B models**
- Achieved through systematic training methodologies

**Key Innovation:**
- Scalable training strategies that could guide future LLM development

---

## Slide 4: Key Contributions

### Three Major Contributions

1. **Model Wind Tunnel Experiments (MWTE)**
   - Use small models to optimize hyperparameters
   - Transfer insights to larger models
   - Systematic approach to training

2. **Warmup-Stable-Decay (WSD) Learning Rate Scheduler**
   - New LR schedule outperforming Cosine scheduler
   - Enables continuous training
   - Dramatic loss reduction in decay phase

3. **Efficient Scaling Law Measurement**
   - Reduced computational cost from O(m²) to O(m)
   - Reveals 192:1 data-to-model ratio (vs Chinchilla's 20:1)
   - Shows small models can absorb far more data

---

## Slide 5: The MiniCPM Family

### Multiple Specialized Variants

| Model | Description | Key Feature |
|-------|------------|-------------|
| **MiniCPM-1.2B** | Base model | Comparable to Llama-7B |
| **MiniCPM-2.4B** | Base model | Comparable to Mistral-7B, Llama-13B |
| **MiniCPM-DPO** | Aligned version | MTBench score 7.25 (beats Llama2-70B!) |
| **MiniCPM-128K** | Long context | 128K context length |
| **MiniCPM-MoE** | Mixture of Experts | 13.6B total, 4B active |

---

## Slide 6: Performance Benchmarks - Base Models

### MiniCPM vs. Larger Models

**MiniCPM-2.4B Performance:**
- **C-Eval**: 51.13% (Chinese knowledge) ✅ #1 in 2B category
- **CMMLU**: 51.07% (Chinese understanding) ✅ #1 in 2B category
- **MMLU**: 53.46% (English knowledge) 🎯 Beats Mistral-7B!
- **HumanEval**: 50.00% (Python coding) 🚀 Outstanding
- **MBPP**: 47.31% (Python problems) 🚀 Outstanding

**Comparison:**
- Outperforms **Llama2-13B** on most tasks
- Matches **Mistral-7B** (which is 3× larger)
- Surpasses **Gemma-7B** on average

---

## Slide 7: Performance Benchmarks - Coding Tasks

### Exceptional Coding Capabilities

**MiniCPM-2.4B Coding Performance:**

| Task | Score | Comparison |
|------|-------|------------|
| **HumanEval** | 50.00% | Higher than Llama2-7B (12.20%) |
| **MBPP** | 47.31% | Exceeds Gemma-7B (50.12%) by 1% |
| **GSM8K** | 53.83% | **#1 SLM** in mathematical reasoning |
| **MATH** | 10.24% | Beats most 7B models |

**Key Insight:** SLMs excel at structured tasks (code, math)

---

## Slide 8: Model Wind Tunnel Experiments

### Systematic Hyperparameter Optimization

**The Aircraft Analogy:**
- Test small-scale models before building full-scale
- Optimize hyperparameters efficiently
- Transfer findings to larger models

**Three Components:**

1. **Scaling Hyperparameters** (Tensor Program)
   - Keep learning rates stable across model sizes
   - Optimal LR ≈ 0.01 for all scales (verified up to 2.1B)

2. **Optimal Batch Size**
   - Systematic exploration of batch vs. loss
   - Found: `bs = 1.21 × 10⁹ × L^(6.24)`
   - Log-linear relationship

3. **Learning Rate Stability**
   - Verified LR consistency using Tensor Program
   - Enables confident scaling

---

## Slide 9: The WSD Learning Rate Scheduler

### Three-Phase Training Strategy

```
Learning Rate
      ↑
 η_max │ ────────────────────── ╲
       │                          ╲
       │                           ╲
       │                            ╲___
       └────────────────────────────────→ Steps
       0   W   T           S

       │   │   │           │
    Warmup│  Stable    Decay
```

**Three Phases:**
1. **Warmup** (0 to W): LR increases **linearly** from 0 to η_max
   - Formula: `LR = (s/W) × η_max`
   - Example: At step 800 with W=1600, LR = 0.5 × max_LR
   
2. **Stable** (W to T): **Constant high LR** for exploration
   - LR remains at η_max throughout
   - Model can make large updates, exploring loss landscape
   
3. **Decay** (T to S): **Rapid exponential decay** for convergence
   - Formula: LR = f(s-T) × η_max
   - Dramatic loss reduction despite smaller LR

**Key Finding:** Loss decreases **dramatically** during decay phase

---

## Slide 9.5: WSD Phase Details

### How Does "Exploration" Work in Stable Phase?

**Warmup Phase (0 to W):**
- Linear increase from 0 to η_max
- You specify **W** (number of warmup steps)
- Typical: W = 1-10% of total training steps
- Example: If max_LR = 0.01 and W = 1600 steps
  - At step 800: LR = (800/1600) × 0.01 = 0.005
  - At step 1600: LR = 0.01

**Stable Phase (W to T):**
- LR = constant η_max (typically 0.01)
- **"Exploration" means:**
  - High LR allows **large parameter updates**
  - Model jumps to different regions of loss landscape
  - Not trapped in local minima
  - Explores globally before fine-tuning

**Decay Phase (T to S):**
- LR decreases exponentially from η_max toward ~0
- MiniCPM uses: `LR = f(s-T) × η_max` where `f(s-T) = 0.5^((s-T)/T)`
- Example: From 0.01 → 0.005 → 0.0025 → ...
- Small steps refine the solution
- Sharp loss reduction!
- Only ~10% of total tokens needed

---

## Slide 10: WSD vs. Cosine Scheduler

### Why WSD is Better

**Cosine LRS:**
- Smooth, continuous decay
- Exploration mixed with convergence
- Hard to reuse checkpoints

**WSD LRS:**
- **Explicit phases** for exploration vs. convergence
- **Checkpoint reusability**: Can experiment with decay schedules
- **10% tokens** needed for decay (vs. 50% for cosine)
- **Sharp loss reduction** during decay phase

**Experimental Evidence:**
- Same performance as cosine
- More efficient (linear vs. quadratic scaling experiments)
- Better for continuous training

---

## Slide 11: Decay Phase Analysis

### Understanding the Magic

**What Happens During Decay?**

**Gradient Statistics Show:**
1. Gradient norms **diminish**
2. Cosine similarity between gradients **becomes positive**
   - Implies consistent parameter updates
3. Loss curvature **increases significantly**
   - Model approaching local optimum

**Key Insight:**
- Despite **smaller weight updates** during decay
- Loss **decreases dramatically**
- Model is "fine-tuning" toward optimum

---

## Slide 12: WSD Advantages

### Continuous Training Made Easy

**Problem:** Traditional training requires restarting from scratch

**WSD Solution:**

1. **Train to stable phase checkpoint** (uses most tokens)
2. **Reuse checkpoint** for different decay schedules
3. **Measure scaling** without full retraining

**Result:**
- **O(m) cost** instead of O(m²)
  - For m model sizes, instead of training m² combinations
  - Reuse stable checkpoints, apply different decays

**Enables:**
- Efficient scaling law exploration
- Experimentation with training schedules
- Continuous training strategies

---

## Slide 13: Scaling Law Findings

### Critical Discovery: 192:1 Data-to-Model Ratio

**Traditional Wisdom (Chinchilla Optimal):**
- 20:1 data-to-model ratio
- "Small models don't need much data"

**MiniCPM Finding:**
- **192:1 data-to-model ratio**
- Nearly **10× more data** than previously believed

**What This Means:**
```
For 1B model: Train on 192B tokens (not 20B)
For 2.4B model: Train on 460B tokens (not 48B)
For 7B model: Train on 1.3T tokens (not 140B)
```

**Key Insight:** Smaller models can absorb **much more data**

---

## Slide 14: Why 192:1 Matters

### Practical Implications

**Training Efficiency:**
- Small models benefit more from data than previously thought
- Better to train 1B model extensively than 3B model on less data

**Deployment Efficiency:**
- Smaller model = **faster inference**
- Smaller model = **less memory required**
- Better for edge devices, mobile, personal computers

**Cost Efficiency:**
- Saves **5×** inference computation
- Only **4×** training compute increase
- Better **inference-compute-optimal** setting

**Example:** 0.036B model matches 0.17B model with 4× training compute, saves 5× inference cost

---

## Slide 15: Scaling Exponents

### Mathematical Analysis

**The Power Law:**
```
L(N,D) = C_N × N^(-α) + C_D × D^(-β) + L_0
```

Where:
- **N** = model size (parameters)
- **D** = data size (tokens)
- **α** = model scaling exponent
- **β** = data scaling exponent

**MiniCPM Findings:**
- **α = 0.29** (model scaling sensitivity)
- **β = 0.23** (data scaling sensitivity)
- **α > β** → slightly emphasize **data scaling**

**Compute Optimal Ratio:**
```
D_opt / N_opt = 192
```

---

## Slide 16: Data-Model Scaling Trade-off

### Understanding the Relationship

**Contour Plot:**
- Shows equal-loss curves
- Model size (N) vs. Data size (D)
- Compute optimal region identified

**Key Observations:**
1. Smaller models are better when compute is low
2. Larger models are better when compute is high
3. Models of different sizes **intersect** at compute-optimal regime
4. This intersection point = 192:1 ratio

**Implication:** 
- At fixed compute budget
- More efficient to use smaller model with more data
- Not larger model with less data

---

## Slide 17: Two-Stage Pre-training Strategy

### Integrating High-Quality Data During Decay

**Observation:** Loss decreases dramatically during decay phase

**Innovation:** Mix high-quality data during decay, not just SFT

**Two Stages:**

1. **Stable Training Phase:**
   - Use large-scale, coarse-quality pre-training data
   - Abundant data supporting continuous training
   - Only requires high learning rate

2. **Decay Phase:**
   - Mix high-quality SFT data with pre-training data
   - More pronounced loss reduction
   - Better alignment with user scenarios

**Result:** 10.5% improvement vs. separate SFT only

---

## Slide 18: Ablation Study Results

### High-Quality Data During Decay

**Experimental Setup:**

**A-1**: 2.4B model, decay using only pre-training data, + 4B token SFT

**A-2**: 2.4B model, decay using **high-quality data mixed**, + 4B token SFT

| Metric | A-1 | A-2 | Improvement |
|--------|-----|-----|-------------|
| C-Eval | 40.0 | 52.6 | **+12.6** ✅ |
| CMMLU | 41.5 | 51.1 | **+9.6** ✅ |
| MATH | 5.1 | 5.4 | +0.3 |

**Conclusion:** Adding high-quality data during decay provides significant benefits

---

## Slide 19: MiniCPM-DPO

### Direct Preference Optimization

**Goal:** Align model with human preferences

**Method:**
- DPO (Direct Preference Optimization) after SFT
- Primary dataset: UltraFeedback
- Proprietary dataset: Enhances code and math capabilities

**Results:**
- **MTBench score**: 7.25 (after DPO)
- Surpasses **Llama2-70B-Chat** (7.24)!
- Despite being 29× smaller in parameters

**Trade-off:**
- Slightly decreased benchmark scores (alignment tax)
- Much better human preference alignment

---

## Slide 20: MiniCPM-128K

### Long Context Processing

**Challenge:** Extend context from 4,096 to 128,000 tokens

**Methods:**
- Adjusted Base Frequency (ABF) for 4K-32K
- NTK-Aware RoPE Scaling and curriculum learning for 32K-128K
- Mix of 44% long data and 56% short data
- Synthetic long QA data

**Results:**
- Comparable to **Mistral-7B-Instruct-128K**
- Outperforms **ChatGLM3-6B-128K** (despite being 2.5× smaller)
- Strong long-context reasoning capabilities

---

## Slide 21: MiniCPM-MoE

### Mixture of Experts Architecture

**Architecture:**
- **Total parameters**: 13.6 billion
- **Active parameters**: 4 billion per token
- **Experts**: 8 experts, 2 activated per token
- Initialization: Sparse Upcycling from dense checkpoint

**Results:**
- Performance on par with **Llama2-34B**!
- Much more efficient (activate only 30% of parameters)
- Demonstrates SLM scalability via MoE

---

## Slide 22: Training Data Distribution

### Diverse Multi-Source Dataset

**Stable Stage Data:**
- CommonCrawl Chinese (25%)
- Dolma (24%)
- CommonCrawl Chinese (25%)
- C4 (15%)
- Pile (8%)
- Code data, papers, math, etc.

**Decay Stage Data:**
- High-quality SFT data mixed
- UltraChat, SlimOrca, EvolInstruct
- Proprietary data (LeetCode, K12 textbooks)
- Maintains diversity while adding quality

**Total:** 1.1 trillion tokens for stable training

---

## Slide 23: Architecture Details

### Model Configuration

**MiniCPM-1.2B:**
- **Parameters**: 1.2B non-embedding (1.4B total)
- **Hidden dimension**: 1,536
- **Layers**: 40 (deep and thin)
- **Attention heads**: 24
- **Context length**: 2,048 tokens

**MiniCPM-2.4B:**
- **Parameters**: 2.4B non-embedding (2.7B total)
- **Hidden dimension**: 2,304
- **Layers**: 40
- **Attention heads**: 36
- **Context length**: 2,048 tokens

**Key Design:** Deep-and-thin architecture for efficiency

---

## Slide 24: Training Loss Curves

### WSD in Action

```
Loss on C4
    ↑
 3.0│
    │   ╱
 2.8│  ╱
    │ ╱  ──────╮
 2.6│╱         ╲___
    └─────────────────→
     200  400  600  800  Tokens (B)
```

**Observations:**
1. **Stable phase**: Gradual loss decrease
2. **Decay phase**: **Sharp loss drop** (orange segment)
3. **SFT stage**: Additional fine-tuning after decay
4. Final checkpoint used from dark green segment

**First drop in 1.2B**: Result of enlarging batch size (similar to decreasing LR)

---

## Slide 25: Evaluation Framework

### Comprehensive Benchmarking

**Evaluation Tool:** UltraEval (open-source)

**Datasets:**
- **Knowledge**: MMLU, C-Eval, CMMLU
- **Coding**: HumanEval, MBPP
- **Math**: GSM8K, MATH
- **Reasoning**: BBH, ARC, HellaSwag

**Methodology:**
- Use best evaluation method per model
- Adapts to different model input-output templates
- Ensures fair comparison

**Frameworks:**
- vLLM for inference acceleration
- Standardized prompts per task

---

## Slide 26: Key Findings - Summary

### What We Learned

1. **SLMs can match LLMs** with proper training strategies
   - 2.4B model rivals 7B-13B models

2. **192:1 data-to-model ratio** is compute-optimal
   - Nearly 10× more data than Chinchilla suggested

3. **WSD scheduler** enables:
   - Continuous training
   - Efficient experimentation
   - Dramatic loss reduction

4. **Small models can absorb** much more data than thought
   - Better inference efficiency
   - Lower deployment costs

---

## Slide 27: Implications for Future Research

### What This Changes

**Training Strategy:**
- Invest in data collection, not just model size
- Use WSD scheduler for efficient experiments
- Focus on specialized variants (MoE, DPO, long-context)

**Deployment:**
- Smaller models more practical for edge devices
- Better inference-compute trade-offs
- Democratize AI access

**Research Direction:**
- Apply WSD to larger LLMs
- Explore further data scaling benefits
- Combine with other optimization techniques

---

## Slide 28: Limitations

### What We Don't Know Yet

1. **LLM Validation Pending**
   - WSD tested only on SLMs (<10B parameters)
   - Not yet validated on larger models (70B+)
   - Remains optimistic about potential benefits

2. **Specific Data Dependencies**
   - Results depend on data quality and diversity
   - Tokenizer choice affects efficiency
   - Data mixture ratios may vary by domain

3. **Reproducibility**
   - Infrastructure differences may affect results
   - Hyperparameter transfers need careful attention

---

## Slide 29: Conclusion

### Key Takeaways

**MiniCPM demonstrates:**
- ✅ **SLMs can rival LLMs** with proper training
- ✅ **Systematic approach** beats trial-and-error
- ✅ **More data beats bigger models** in many scenarios
- ✅ **Efficient training strategies** enable rapid experimentation

**Impact:**
- Democratizes AI research (smaller models accessible)
- Improves deployment efficiency (edge-friendly)
- Guides future LLM development strategies

**Future Directions:**
- Apply to larger models
- Explore further scaling benefits
- Combine with retrieval augmentation

---

## Slide 30: Resources & Links

### Implementation Details

**Pre-trained Models:**
- GitHub: `https://github.com/OpenBMB/MiniCPM`
- Hugging Face: Available for download

**Evaluation Tools:**
- UltraEval: `https://ultraeval.openbmb.cn/home`

**Training Code:**
- PyTorch implementation
- WSD scheduler available
- Reproducible experiments

**Papers:**
- arXiv: 2404.06395
- Technical blog: February 1, 2024
- Open-source community integration

---

## Slide 31: Questions & Discussion

### Thank You!

**Key Insights to Remember:**
1. Small models + smart training = big results
2. WSD scheduler enables efficient experimentation
3. 192:1 ratio reveals data's importance
4. SLMs are viable alternatives to LLMs

**Questions?**

**Contact:**
- Isaac Kobby Anni
- Department of Computer Science
- Preliminary Exam Presentation

