# Lab 3: Contextual Bandit-Based News Article Recommendation System

**Course:** Reinforcement Learning Fundamentals  
**Student Name:** Dipti Dhawade  
**Roll Number:** U20230146  
**GitHub Branch:** dipti_U20230146

---

## Table of Contents
1. [Introduction](#introduction)
2. [Implementation Overview](#implementation-overview)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Analysis and Discussion](#analysis-and-discussion)
6. [Conclusions](#conclusions)

---

## Introduction

This project implements a **Contextual Multi-Armed Bandit (CMAB)** system for personalized news article recommendations. The system learns to recommend optimal news categories based on user profiles, treating user types as contexts and news categories as arms.

### Problem Formulation
- **Contexts**: 3 user types (User1, User2, User3)
- **Arms**: 4 news categories per context (Entertainment, Education, Tech, Crime)
- **Total Arms**: 12 (3 contexts × 4 categories)
- **Objective**: Maximize user engagement by learning optimal news category recommendations for each user type

### Arm Index Mapping
| Arm Index | News Category | User Context |
|-----------|---------------|--------------|
| 0-3       | Entertainment, Education, Tech, Crime | User1 |
| 4-7       | Entertainment, Education, Tech, Crime | User2 |
| 8-11      | Entertainment, Education, Tech, Crime | User3 |

---

## Implementation Overview

### 1. Data Preprocessing (10 Points)
- **Datasets Used**:
  - `train_users.csv`: 2000 users for training/validation (80/20 split)
  - `test_users.csv`: 2000 users for testing
  - `news_articles.csv`: Collection of news articles across 4 categories

- **Preprocessing Steps**:
  - Handled missing values in the `age` column using median imputation
  - Encoded categorical features (`browser_version`, `region_code`) using LabelEncoder
  - Handled unseen categories in test data by assigning them to -1
  - Converted boolean features to integers
  - Applied stratified train-test split (80/20) for validation

**Dataset Shapes**:
- Training set: 1,600 samples × 31 features
- Validation set: 400 samples × 31 features
- Test set: 2,000 samples × 31 features

---

### 2. User Classification (10 Points)

**Model**: Gradient Boosting Classifier  
**Purpose**: Serves as the "Context Detector" to classify users into User1, User2, or User3

**Performance Metrics**:
```
Validation Accuracy: 90.50%

Classification Report:
              precision    recall  f1-score   support
    user_1       0.95      0.88      0.92       142
    user_2       0.89      0.94      0.91       125
    user_3       0.87      0.88      0.88       133

  accuracy                           0.90       400
 macro avg       0.90      0.90      0.90       400
weighted avg     0.91      0.90      0.90       400

Confusion Matrix:
[[ 125   14    3]
 [   6  117    2]
 [  14    0  119]]
```

**Top 5 Most Important Features**:
1. `session_duration` (39.31%)
2. `region_code` (33.33%)
3. `age` (5.54%)
4. `preferred_price_range` (2.04%)
5. `content_variety` (1.76%)

**Test Set Predictions**:
- User1: 674 users (33.70%)
- User2: 710 users (35.50%)
- User3: 616 users (30.80%)

---

### 3. Contextual Bandit Algorithms (45 Points)

All three algorithms were implemented with **separate models for each user context**, ensuring proper contextual learning.

#### 3.1 Epsilon-Greedy Strategy (15 Points)

**Implementation**:
- Three separate bandit instances (one per user context)
- Epsilon-greedy exploration strategy
- Incremental Q-value updates: `Q(a) ← Q(a) + (r - Q(a))/N(a)`

**Hyperparameter Tuning** (ε values tested):
```
ε = 0.01: Final Average Reward = 4.9464
ε = 0.10: Final Average Reward = 4.4459
ε = 0.30: Final Average Reward = 3.3814
```

**Expected Reward Distribution** (ε = 0.01):
| User Context | Entertainment | Education | Tech | Crime |
|--------------|---------------|-----------|------|-------|
| User1        | 4.6989        | -6.9429   | **5.2523** | 4.7208 |
| User2        | **5.2823**    | -2.7791   | -4.9813 | -3.9341 |
| User3        | -8.8180       | **4.4917** | -0.8244 | -1.4893 |

**Best News Category per User**:
- User1: **Tech** (Q = 5.2523)
- User2: **Entertainment** (Q = 5.2823)
- User3: **Education** (Q = 4.4917)

**Observations**:
- Lower ε (0.01) achieves highest final reward through aggressive exploitation
- Higher ε (0.3) explores more but converges to lower reward
- ε = 0.1 provides balanced exploration-exploitation

---

#### 3.2 Upper Confidence Bound (UCB) Strategy (15 Points)

**Implementation**:
- Three separate UCB bandit instances
- UCB formula: `Q(a) + c × √(ln(t) / N(a))`
- Systematic exploration based on uncertainty

**Hyperparameter Tuning** (C values tested):
```
C = 0.5: Final Average Reward = 4.9696
C = 1.0: Final Average Reward = 4.9861 ⭐ BEST
C = 2.0: Final Average Reward = 4.9850
```

**Expected Reward Distribution** (C = 1.0):
| User Context | Entertainment | Education | Tech | Crime |
|--------------|---------------|-----------|------|-------|
| User1        | 4.6916        | -6.9513   | **5.2589** | 4.7120 |
| User2        | **5.2831**    | -2.8061   | -5.0222 | -3.9381 |
| User3        | -8.8142       | **4.4910** | -0.7856 | -1.4804 |

**Best News Category per User**:
- User1: **Tech** (Q = 5.2589)
- User2: **Entertainment** (Q = 5.2831)
- User3: **Education** (Q = 4.4910)

**Observations**:
- UCB achieves the highest overall performance (4.9861)
- C = 1.0 provides optimal balance
- More stable convergence compared to epsilon-greedy
- Eliminates random exploration in favor of principled uncertainty-based exploration

---

#### 3.3 SoftMax Strategy (15 Points)

**Implementation**:
- Three separate SoftMax bandit instances
- Temperature parameter: τ = 1.0 (fixed)
- Probability: `P(a) = exp(Q(a)/τ) / Σ exp(Q(i)/τ)`

**Performance**:
```
τ = 1.0: Final Average Reward = 4.8923
```

**Expected Reward Distribution** (τ = 1.0):
| User Context | Entertainment | Education | Tech | Crime |
|--------------|---------------|-----------|------|-------|
| User1        | 4.6972        | -6.9550   | **5.2519** | 4.6981 |
| User2        | **5.2771**    | -2.7772   | -5.0044 | -3.9391 |
| User3        | -8.8209       | **4.4823** | -0.7977 | -1.4927 |

**Best News Category per User**:
- User1: **Tech** (Q = 5.2519)
- User2: **Entertainment** (Q = 5.2771)
- User3: **Education** (Q = 4.4823)

**Observations**:
- Competitive performance with epsilon-greedy
- Smooth probability distribution prevents abrupt switches
- Temperature τ = 1.0 provides reasonable exploration
- More gradual convergence compared to UCB

---

### 4. Recommendation Engine (20 Points)

**End-to-End Workflow**:
1. **Classify User**: Predict user category (User1/User2/User3) using Gradient Boosting Classifier
2. **Select Category**: Use trained bandit policy to select optimal news category
3. **Recommend Article**: Randomly sample a specific article from the selected category
4. **Output**: User category + recommended news category + article details

**Implementation**:
```python
class NewsRecommendationEngineImproved:
    def recommend(self, user_features):
        # Step 1: Classify user (get context)
        user_category = classify_user(user_features)
        
        # Step 2: Get bandit for this context
        bandit = context_bandits[user_category]
        
        # Step 3: Select best arm (news category)
        best_arm = argmax(bandit.Q)
        news_category = arm_to_category[best_arm]
        
        # Step 4: Sample article from category
        article = sample_article(news_category)
        
        return {user_category, news_category, article}
```

**Test Set Recommendations** (2000 users):
- Entertainment: 710 recommendations (35.50%)
- Tech: 674 recommendations (33.70%)
- Education: 616 recommendations (30.80%)
- Crime: 0 recommendations (0.00%)

**Observations**:
- Crime category receives no recommendations (consistently lowest Q-values)
- Distribution aligns with predicted user categories
- System successfully personalizes recommendations per user type

---

## Results

### Performance Comparison

| Strategy | Hyperparameter | Final Avg Reward | Rank |
|----------|---------------|------------------|------|
| UCB | C = 1.0 | **4.9861** | 🥇 1st |
| UCB | C = 2.0 | 4.9850 | 2nd |
| UCB | C = 0.5 | 4.9696 | 3rd |
| Epsilon-Greedy | ε = 0.01 | 4.9464 | 4th |
| SoftMax | τ = 1.0 | 4.8923 | 5th |
| Epsilon-Greedy | ε = 0.1 | 4.4459 | 6th |
| Epsilon-Greedy | ε = 0.3 | 3.3814 | 7th |

**Winner**: UCB with C = 1.0 (Average Reward: 4.9861)

---

### Visualizations

#### 1. Average Reward vs Time
All three strategies show convergence around 5000-7000 steps:
- **UCB**: Fastest and most stable convergence
- **Epsilon-Greedy (ε=0.01)**: Quick convergence but slightly lower plateau
- **SoftMax (τ=1.0)**: Smooth, gradual convergence

#### 2. Hyperparameter Sensitivity

**Epsilon-Greedy**:
- ε = 0.01: Aggressive exploitation → Higher reward
- ε = 0.1: Balanced approach → Moderate reward
- ε = 0.3: Over-exploration → Lower reward

**UCB**:
- C = 0.5: Conservative exploration → Good performance
- C = 1.0: Optimal balance → Best performance
- C = 2.0: Aggressive exploration → Slightly lower than C=1.0

#### 3. Context-Specific Performance
Separate models per context show distinct learning patterns:
- Each user type exhibits different reward distributions
- Context-specific learning is crucial for personalization
- All contexts converge to stable policies

---

## Analysis and Discussion

### 1. Algorithm Comparison

#### UCB (Winner)
**Strengths**:
- Systematic, uncertainty-driven exploration
- Optimal theoretical guarantees
- Fastest convergence to best arms
- No random exploration waste

**Weaknesses**:
- Requires careful tuning of C parameter
- Can be overly optimistic in early stages

#### Epsilon-Greedy
**Strengths**:
- Simple to implement and understand
- Low computational overhead
- Tunable exploration rate

**Weaknesses**:
- Random exploration wastes trials
- Continues exploring even after convergence
- Sensitive to epsilon choice

#### SoftMax
**Strengths**:
- Probabilistic exploration
- Smooth transitions between arms
- Never completely ignores any arm

**Weaknesses**:
- Requires temperature tuning
- Can be slow to converge
- May over-explore good (but not best) arms

---

### 2. Contextual Learning Insights

**User1 Preferences**:
- Strongly prefers **Tech** content
- Avoids **Education** content (highly negative rewards)
- Clear preference hierarchy: Tech > Crime > Entertainment

**User2 Preferences**:
- Strongly prefers **Entertainment** content
- Dislikes Tech, Crime, and Education
- Most concentrated preference (one dominant category)

**User3 Preferences**:
- Prefers **Education** content
- Strongly dislikes Entertainment
- Moderate aversion to Tech and Crime

**Key Insight**: Contextual information (user type) is crucial — the same news category (e.g., Tech) yields vastly different rewards depending on the user context.

---

### 3. Hyperparameter Sensitivity

**Epsilon (ε) in Epsilon-Greedy**:
- Optimal range: 0.01 - 0.05 for exploitation-focused scenarios
- ε = 0.1 acceptable for balanced exploration
- ε > 0.2 leads to significant performance degradation

**C in UCB**:
- Sweet spot: C = 1.0 - 1.5
- Lower C (0.5) underexplores
- Higher C (2.0) overexplores without much benefit

**Temperature (τ) in SoftMax**:
- τ = 1.0 provides reasonable performance
- Further tuning of τ could improve results

---

### 4. Convergence Analysis

**Time to Convergence**:
- UCB: ~3,000 - 4,000 steps
- Epsilon-Greedy (ε=0.01): ~4,000 - 5,000 steps
- SoftMax (τ=1.0): ~5,000 - 6,000 steps

**Stability**:
- All algorithms achieve stable policies by step 7,000
- UCB shows least variance after convergence
- Epsilon-greedy exhibits slight oscillations due to random exploration

---

## Conclusions

### Main Findings
1. **UCB outperforms** other strategies (4.9861 avg reward)
2. **Context is crucial**: Same news category yields different rewards per user type
3. **Hyperparameter tuning matters**: ε and C significantly impact performance
4. **Separate models per context** enable proper contextual learning
5. **Convergence achieved** within 5,000-7,000 steps for all algorithms

### Learned Patterns
- **User1**: Tech enthusiasts who avoid Education content
- **User2**: Entertainment seekers who dislike all other categories
- **User3**: Education-focused users who avoid Entertainment

---

## Technical Details

### Environment
- Python 3.13.5
- Key Libraries: numpy, pandas, scikit-learn, matplotlib
- Reward Sampler: `rlcmab-sampler` (Roll Number: 146)

### File Structure
```
.
├── lab3_results_U20230146.ipynb  # Main implementation notebook
├── README.md                            # This file
├── data/
│   ├── train_users.csv
│   ├── test_users.csv
│   └── news_articles.csv
└── outputs/
    ├── test_predictions.csv             # User classifications
    ├── test_recommendations.csv         # Final recommendations
```

### Reproducibility
All experiments use fixed random seed where applicable. Results may vary slightly due to:
- Random context selection during training
- Random article sampling from categories
- Stochastic nature of bandit algorithms

---

## Appendix: Code Snippets

### Epsilon-Greedy Implementation
```python
class EpsilonGreedyBanditSingle:
    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_arms)  # Explore
        else:
            return np.argmax(self.Q)  # Exploit
```

### UCB Implementation
```python
class UCBBanditSingle:
    def select_arm(self):
        if self.t <= self.n_arms:
            return self.t - 1  # Initialize
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / (self.N + 1e-5))
        return np.argmax(ucb_values)
```

### SoftMax Implementation
```python
class SoftmaxBanditSingle:
    def select_arm(self):
        exp_values = np.exp(self.Q / self.tau)
        probs = exp_values / np.sum(exp_values)
        return np.random.choice(self.n_arms, p=probs)
```

### Arm Index Mapping
```python
def get_arm_index(user_category, news_category):
    context_offset = user_to_context[user_category] * 4
    arm_offset = category_to_arm[news_category]
    return context_offset + arm_offset
```

---