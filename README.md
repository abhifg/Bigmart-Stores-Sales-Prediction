## 🛒 Bigmart Store Sales Prediction

A machine learning web application that predicts item-outlet sales for Bigmart stores. Built with Python/Flask, containerized with Docker, and deployed on **Microsoft Azure App Service**.

---

## 📌 Problem Statement

Shopping malls and Big Marts track individual item sales data to forecast future client demand and adjust inventory management. This project builds an ML model to predict the sales of different stores of BigMart based on product and outlet features.

---

## 🛠 Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| Web Framework | Flask |
| ML Libraries | Scikit-learn, XGBoost, NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Containerization | Docker |
| Cloud Platform | Microsoft Azure |
| Container Registry | Azure Container Registry (ACR) |
| Hosting | Azure App Service |
| Authentication | Azure Managed Identity |

---

## 📊 Dataset

**Source:** [Kaggle - BigMart Sales Dataset](https://www.kaggle.com/datasets/ahmadrezagholami2001/bigmart-sales-dataset)

| Feature | Description |
|---|---|
| Item_Weight | Weight of the product |
| Item_Fat_Content | Whether the product is low fat or regular |
| Item_Visibility | % of display area allocated to the product in the store |
| Item_Type | Category to which the product belongs |
| Item_MRP | Maximum Retail Price of the product |
| Outlet_Size | Size of the store (Small / Medium / High) |
| Outlet_Location_Type | Type of city (Tier 1 / Tier 2 / Tier 3) |
| Outlet_Type | Grocery Store or Supermarket type |
| Outlet_Establishment_Year | Year in which the store was established |
| **Item_Outlet_Sales** | **Target variable — sales of the product** |

---

## 🔍 Exploratory Data Analysis

Key insights discovered during EDA:

- **Item Visibility** is right-skewed with many outliers
- **Item Fat Content** had redundant classes (`LF`, `low fat`, `reg`) which were standardized
- **Item MRP** has a strong positive correlation with Item Outlet Sales
- **Supermarket Type 3** outlets have the highest average sales
- Most outlets in Tier 3 cities; Tier 1 cities only have Supermarket Type 1 and Grocery Stores
- No multicollinearity found among numerical features

---

## ⚙️ Feature Engineering

### 1. Outlier Removal
Removed outliers in `Item_Visibility` using IQR method:
```
Lower Fence = Q1 - 1.5 * IQR
Upper Fence = Q3 + 1.5 * IQR
```

### 2. Missing Value Imputation
- **Item_Weight** — imputed with mean weight of the same `Item_Type` category
- **Outlet_Size** — imputed based on `Outlet_Type`:
  - Supermarket Type 1 → Small
  - Supermarket Type 2 → Medium
  - Supermarket Type 3 → Medium
  - Grocery Store → Small
- **Item_Visibility = 0** — replaced with mean visibility of the same `Outlet_Type`

### 3. Encoding Categorical Variables
- **One Hot Encoding** → `Item_Fat_Content`, `Outlet_Type`, `Item_Type`
- **Ordinal Encoding** → `Outlet_Size` (Small=0, Medium=1, High=2)
- **Ordinal Encoding** → `Outlet_Location_Type` (Tier 1=0, Tier 2=1, Tier 3=2)

### 4. Transformations for Gaussian Distribution
- **Item_Visibility** → Cube root transform: `visibility^(1/3)`
- **Item_Outlet_Sales** → Power transform: `sales^(1/8)`

### 5. Feature Addition
- `Outlet_Establishment_Age = 2026 - Outlet_Establishment_Year`

### 6. Scaling
- `StandardScaler` applied to all numerical features

---

## 🤖 Model Selection

Trained and evaluated multiple regression models using cross-validation:

| Model | Technique | Best R² Score |
|---|---|---|
| Linear Regression | Cross Validation (cv=5) | Baseline |
| Lasso Regression | GridSearchCV (cv=5) | Tuned |
| Ridge Regression | GridSearchCV (cv=5) | Tuned |
| Random Forest | RandomizedSearchCV (n_iter=100) | **~71.37%** ✅ |
| XGBoost | RandomizedSearchCV (cv=8, n_iter=100) | Compared |

**Winner: Random Forest Regressor** with the highest R² score of ~71.37%

Best Parameters:
```
n_estimators: 22, max_depth: 5, min_samples_split: 10, min_samples_leaf: 22
```

---

## 💻 Local Setup

```bash
# Clone the repository
git clone https://github.com//Bigmart-Stores-Sales-Prediction.git
cd Bigmart-Stores-Sales-Prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Visit `http://localhost:80` in your browser.

---

## ☁️ Azure Deployment

The app is containerized using Docker and deployed to **Azure App Service** via **Azure Container Registry (ACR)**.

### Architecture

```
Local Machine (Mac M1/M2)
    │
    │  docker buildx build --platform linux/amd64
    ▼
Azure Container Registry (ACR)
    │
    │  Managed Identity (AcrPull Role)
    ▼
Azure App Service (Linux Container)
```

**Key implementation details:**
- Built Docker image for `linux/amd64` platform (required for Azure when developing on Mac M1/M2)
- Configured **Managed Identity** for secure, credential-free authentication between App Service and ACR
- Resolved cross-platform architecture incompatibility (ARM64 → AMD64)

### Deployment Screenshot
**Home Page**
<img width="2940" height="1912" alt="image" src="https://github.com/user-attachments/assets/4b2322a6-b202-40ed-a970-f9de357c2129" />

**Prediction Form**
<img width="2940" height="1912" alt="image" src="https://github.com/user-attachments/assets/98ba1827-9270-4812-9204-62bd9106904b" />

**Prediction**
<img width="2940" height="1912" alt="image" src="https://github.com/user-attachments/assets/30ad69db-3a5d-4c83-a745-6f7d12229ed8" />




---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
