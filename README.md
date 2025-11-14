**When Nostalgia Backfires**

Statistical methodology for analyzing context-dependent effects of nostalgic political advertising on voter turnout across 393 battleground counties (2020-2024)

**Statistical Equations**

**Primary Analysis**

Spearman Rank Correlation

ρ = 1 - (6Σd²) / [n(n² - 1)]

OLS Regression (Simple)

ΔTurnout = β₀ + β₁(ΔNostalgia) + ε

OLS Regression (With Controls)

ΔTurnout = β₀ + β₁(ΔNostalgia) + β₂(Income) + β₃(Education) + ε

Standardized OLS

z(ΔTurnout) = β₀ + β₁·z(ΔNostalgia) + β₂·z(Income) + β₃·z(Education) + ε | where: z(X) = (X - μₓ) / σₓ

HC3 Robust Standard Errors SE(β̂) = √[(X'X)⁻¹X'ΩX(X'X)⁻¹] | where: Ω = diag[ε̂ᵢ²/(1-hᵢᵢ)²]

**Robustness Checks**

Bootstrap Confidence Intervals (5,000 iterations)

For b = 1 to 5,000:
  1. Sample n observations with replacement
  2. Calculate ρ*ᵦ = Spearman(ΔNostalgia*, ΔTurnout*)
  
95% CI = [P₂.₅(ρ*), P₉₇.₅(ρ*)]

Permutation Tests (5,000 iterations)

For p = 1 to 5,000:
  1. Randomly shuffle ΔNostalgia across counties
  2. Calculate ρₚ = Spearman(ΔNostalgia_permuted, ΔTurnout)
  
p-value = [Σ I(|ρₚ| ≥ |ρ_obs|)] / 5,000

Variance Inflation Factor (VIF)

For each predictor Xⱼ: VIFⱼ = 1 / (1 - R²ⱼ) | where R²ⱼ from auxiliary regression: Xⱼ = α₀ + Σₖ≠ⱼ αₖXₖ + εⱼ

State-Level Subgroup Analysis

For each state s: ρₛ = Spearman(ΔNostalgiaₛ, ΔTurnoutₛ); 95% CIₛ = [P₂.₅(ρ*ₛ), P₉₇.₅(ρ*ₛ)] [via bootstrap]

Fisher r-to-z Transformation

Step 1: Transform correlations to z-scores
z₁ = 0.5 × ln[(1 + ρ₁)/(1 - ρ₁)]
z₂ = 0.5 × ln[(1 + ρ₂)/(1 - ρ₂)]

Step 2: Calculate standard error
SE = √[1/(n₁-3) + 1/(n₂-3)]

Step 3: Test statistic
Z = (z₁ - z₂) / SE

Step 4: P-value (two-tailed)
p = 2 × [1 - Φ(|Z|)]

Demographic Subgroup Analysis

Split at median: Median(% White) = 64.8%
Group 1: Counties where % White > 64.8%
Group 2: Counties where % White ≤ 64.8%

For each group:
ρ_high = Spearman(ΔNostalgia_high, ΔTurnout_high)
ρ_low = Spearman(ΔNostalgia_low, ΔTurnout_low)

Compare groups:
z_high = 0.5 × ln[(1 + ρ_high)/(1 - ρ_high)]
z_low = 0.5 × ln[(1 + ρ_low)/(1 - ρ_low)]

SE = √[1/(n_high-3) + 1/(n_low-3)]

Z = (z_high - z_low) / SE
