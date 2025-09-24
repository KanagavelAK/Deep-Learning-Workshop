## Binary Classification with Neural Networks on the Census Income Dataset
This project builds a binary classification model using PyTorch to predict whether an individual earns more than $50,000 annually, based on the Census Income Dataset (income.csv) containing 30,000 entries.

### Task Description
Dataset: income.csv (30,000 entries).

Goal: Predict income level (<=50K or >50K).

## Data Preparation

#### Separate columns:

1. Categorical features 
        
2. Continuous features    
        
3. Label column (income).
        
#### Transform data:

4. Convert categorical values to category codes.
        
5. Standardize continuous features.
        
6. Encode labels (0 = <=50K, 1 = >50K).
        
#### Create tensors:

7. Categorical values → LongTensors.
        
8. Continuous values → FloatTensors.
        
9. Labels → LongTensors.
        
#### Split dataset:

-- Training set: 25,000 samples.
        
-- Test set: 5,000 samples.
        

## Model Design

#### TabularModel Class:

1. Embeddings for categorical features.
       
2. BatchNorm for continuous features.
       
3. One hidden layer with 50 neurons.
       
4. Dropout (p=0.4) for regularization.
       
5. Output layer with 2 classes (<=50K, >50K).

## Training

1. Random seed set for reproducibility.

2. Loss function: CrossEntropyLoss.

3. Optimizer: Adam (lr=0.001).

4. Epochs: 300.

5. Training loop: forward pass → loss calculation → backward pass → optimizer step.

## Evaluation

1. Evaluate on the test set (5,000 samples).

2. Report:

   Test Loss
   
   Accuracy (%)

## How to Run

1. Install dependencies:

              ```pip install torch pandas scikit-learn```

2. Place income.csv in the working directory.

3. Run training script:
   
              ```python train.py```

5. After training, the script prints test loss and accuracy.

6. Final accuracy ~ 80–85% (depending on random seed and preprocessing).
