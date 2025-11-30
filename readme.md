## Model Whisper: Steering Vectors Unlock  Large Language Models’ Potential in Test-time

## Code Status

- **Testing Code**: This repository currently provides the code for evaluating our proposed method.
- **Training Code**: The full training code and associated scripts will be made publicly available upon the acceptance of our paper. We are committed to releasing all necessary components to ensure complete reproducibility.

## Reproducibility

To facilitate the verification of our results, we provide a TTSV checkpoint.

### TTSV Checkpoint

We have included the TTSV weights for the **Qwen2.5-math-7b** model, trained on the **MATH500** dataset. You can find this checkpoint at: checkpoints/qwen2.5-math-7b_math500/

### Model Weights

Due to file size limitations for supplementary materials, the base model weights for **Qwen2.5-math-7b** are not included in this repository. Please download the model from its official source (e.g., Hugging Face) and place it in the appropriate directory as required by the evaluation scripts.

## Evaluation

To run the evaluation and reproduce the results on the corresponding test sets, please follow these steps:

* **Create and activate a virtual environment**: 

  ```bash
  conda create -n TTSV python=3.10
  conda activate TTSV
  ```

* **Install pip dependencies**

  ```bash
  bash install.sh
  ```


* **Download Model**: Download the Qwen2.5-math-7b model and place it in the designated path:

  ```txt
  model/Qwen2.5-Math-7B
  ```

* **Run Evaluation Script**: Execute the following command from the root directory of this project:

  ```bash
  bash Qwen2.5-Eval/evaluation/sh/eval_all_math.sh
  ```