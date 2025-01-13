## **1. Setting Up the Server and Environment**

### **1.1 Log in to the Server**
Log into your remote server using SSH (PowerShell for Windows):

```bash
ssh <username>@<server-address>
```

---

### **1.2 Transfer Files to the Server**

Use `scp` to securely transfer the project files (e.g., `Sindhu_Pasupuleti.zip`) to the server:

```bash
scp /path/to/Sindhu_Pasupuleti.zip <username>@<server-address>:/path/to/destination
```

Once transferred, unzip the file:

```bash
unzip Sindhu_Pasupuleti.zip
```

The directory structure will look like this:

```plaintext
├── Model.ipynb                        # Notebook for main model development
├── Dataset.ipynb                      # Notebook for dataset creation and baseline model
├── Bias Detection.ipynb	       # Notebook for detecting multi-dimensional bias
├── requirements.txt                   # Dependencies list
├── balanced_biased_resume_dataset.csv # Custom dataset
└── best_bertm_full.pth                # Trained BERT model
```

---

### **1.3 Set Up the Environment**

1. **Activate the Conda Environment**:

   Make sure Conda is installed, then activate the required environment:

   ```bash
   conda activate <environment_name>
   ```

2. **Install Dependencies**:

   Install all required libraries from the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

---

## **2. Dataset Generation Workflow**

### **2.1 Overview of `Dataset.ipynb`**

This notebook generates a synthetic dataset of resumes with demographic attributes and introduces bias based on predefined probabilities.

---

### **2.2 Steps to Run `Dataset.ipynb`**

1. **Install Required Libraries**:

   Ensure `pandas` and `numpy` are installed:

   ```bash
   pip install pandas numpy
   ```

2. **Run the Notebook**:

   Open the notebook using Jupyter and execute all cells:

   ```bash
   jupyter notebook
   ```

   - Navigate to `Dataset.ipynb` and run all cells sequentially.

3. **Generated Dataset**:

   The final balanced dataset is saved as `balanced_biased_resume_dataset.csv` with the following structure:

   | Name  | Gender  | Age | Race    | Resume Text      | Shortlisted |
   |-------|---------|-----|---------|------------------|-------------|
   | John Doe | Male    | 25  | White   | Professional experience... | 1           |

---

## **3. Model Training Workflow**

### **3.1 Overview of `Model.ipynb`**

This notebook trains a **BERT-based classifier** to predict resume shortlisting. It includes:

1. **Data Preprocessing**: Loading and tokenizing text data.
2. **Model Training**: BERT fine-tuning using PyTorch.
3. **Bias Mitigation**: Training a GAN to balance demographic features.
4. **Evaluation**: Measuring model performance (accuracy, precision, recall, ROC-AUC).

---

### **3.2 Steps to Run `Model.ipynb`**

1. **Load the Notebook**:

   Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   - Open `Model.ipynb`.

2. **Execute All Cells**:

   Follow the workflow to perform:

   - **Data Preprocessing**:
     - Tokenize text data using BERT tokenizer.
     - Normalize and encode demographic features.

   - **Model Training**:
     - Train the BERT classifier using cross-entropy loss and early stopping.
     - Monitor training using TensorBoard:

       ```bash
       tensorboard --logdir=runs
       ```

   - **GAN Training**:
     - Generate synthetic demographics for bias balancing.

   - **Evaluation**:
     - Save the best BERT model as `best_bertm_full.pth`.

3. **Outputs**:

   - **Model**: Saved as `best_bertm_full.pth`.
   - **Metrics**: Precision, recall, F1-score, and ROC-AUC.
   - **GAN Visualizations**: Histograms comparing real vs. synthetic demographics.

---

## **4. Bias Detection Workflow**

### **4.1 Overview of Bias Detection**

The bias detection workflow evaluates fairness metrics and computes demographic disparities in the BERT model predictions.

Metrics include:

- **Disparate Impact (DI)**: Measures selection rate differences between groups.
- **Equal Opportunity (EO)**: Compares True Positive Rates (TPR) across groups.
- **General Fairness Metrics**: Precision, recall, and fairness-based evaluations.

---

### **4.2 Steps to Run Bias Detection**

1. **Load the Notebook**:

   Open the `Bias Detection.ipynb` file:

   ```bash
   jupyter notebook
   ```

2. **Execute All Cells**:

   Key steps include:

   - **Data Loading**:
     Load the dataset (`balanced_biased_resume_dataset.csv`) and pre-trained BERT model.

   - **Bias Metrics Calculation**:
     - Disparate Impact (DI)
     - True Positive Rates (TPR)
     - Equal Opportunity (EO)

   - **Visualization**:
     Generate bar plots and histograms to visualize fairness metrics.

   - **GAN Perturbation**:
     Use a trained GAN to perturb demographic features and recalculate fairness metrics.

3. **Outputs**:

   - Bias metrics (DI, EO, TPR) for all demographic groups.
   - Visualizations comparing fairness metrics across groups.
   - GAN-generated demographic perturbations for fairness analysis.

---

## **5. Outputs and File Locations**

After executing all notebooks:

| **File**                             | **Description**                                      |
|--------------------------------------|------------------------------------------------------|
| `balanced_biased_resume_dataset.csv` | Final generated dataset with biases.                 |
| `best_bertm_full.pth`                | Trained BERT classification model.                  |
| Bias metrics visualizations          | Visual outputs for disparate impact and fairness.    |
| TensorBoard logs                     | Logs for monitoring training metrics and loss curves.|

---

## **6. Monitoring Training**

To monitor model training using TensorBoard:

1. Run TensorBoard:

   ```bash
   tensorboard --logdir=runs
   ```

2. Open the link displayed in the terminal (e.g., `http://localhost:8882`).

---

## **7. Dependencies**

The project requires the following libraries:

- `pandas`
- `numpy`
- `torch`
- `transformers`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## **8. Conclusion**

This workflow enables:

1. **Synthetic Dataset Generation** with controlled demographic biases.
2. **BERT-Based Model Training** for resume shortlisting.
3. **Bias Detection** using fairness metrics.
4. **Bias Mitigation** through GAN-based demographic balancing.
