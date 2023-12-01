import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv("clean_dataset.csv")
print(data.head())
features = [
    "Gender",
    "Age",
    "Debt",
    "Married",
    "BankCustomer",
    # "EducationLevel",
    "Industry",
    "Ethnicity",
    "YearsEmployed",
    "PriorDefault",
    "Employed",
    "CreditScore",
    "DriversLicense",
    "Citizen",
    "ZipCode",
    "Income",
    "Approved",
]


# encode categorical features
def encode_labels(df):
    for c in df.columns:
        if df[c].dtype == "object":
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df


def eda(data):
    df = data.copy()
    print(df.head())
    print(df.shape)
    print(data.dtypes)

    sns.countplot(x="Approved", data=df)
    plt.show()

    sns.countplot(x="Gender", data=df)
    plt.show()

    sns.countplot(x="Ethnicity", data=df)
    plt.show()

    df = df.replace("?", np.nan)
    df["Age"] = df["Age"].astype(float)

    sns.pairplot(
        df,
        vars=["Age", "Debt", "Income", "YearsEmployed", "CreditScore"],
        hue="Approved",
    )
    plt.show()

    df = encode_labels(df)
    df = df[
        [
            "Approved",
            "Gender",
            "Age",
            "Debt",
            "Income",
            "YearsEmployed",
            "CreditScore",
            "PriorDefault",
            "Employed",
            "DriversLicense",
            "Married",
        ]
    ]

    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.show()


eda(data)


def data_prep(data):
    data = data.replace("?", np.nan)
    data["Age"] = data["Age"].astype(float)
    # drop rows with missing values
    data = data.dropna()

    data = encode_labels(data)

    trainX, testX, trainY, testY = train_test_split(
        data.drop(["Approved"], axis=1),
        data["Approved"],
        test_size=0.2,
        random_state=42,
    )

    # scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)

    return trainX, testX, trainY, testY


# trainX, testX, trainY, testY = data_prep(data)


def logistic_regression(trainX, testX, trainY, testY):
    clf = LogisticRegression(random_state=42)
    clf.fit(trainX, trainY)
    predY = clf.predict(testX)
    print(accuracy_score(testY, predY))
    print(confusion_matrix(testY, predY))
    # save the model
    with open("logistic.pkl", "wb") as f:
        pickle.dump(clf, f)


# logistic_regression(trainX, testX, trainY, testY)


def view_model():
    with open("logistic.pkl", "rb") as f:
        clf = pickle.load(f)
    print(clf.coef_)
    print(clf.intercept_)

    coefs_df = pd.DataFrame({"Features": features[:-1], "Coefficients": clf.coef_[0]})
    sns.barplot(
        x="Coefficients",
        y="Features",
        data=coefs_df,
        order=coefs_df.sort_values("Coefficients").Features,
    )
    plt.xlabel("Coefficient value")
    plt.ylabel("Feature")
    plt.title("Coefficients of Logistic Regression Model")
    # plt.axvline(x=0, color="gray", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.show()
    return clf


# view_model()


def chi_squared(feature="Ethnicity"):
    label_encoder = LabelEncoder()
    data["Ethnicity_encoded"] = label_encoder.fit_transform(data[feature])
    data["approval_encoded"] = label_encoder.fit_transform(data["Approved"])
    contingency_table = pd.crosstab(data["Ethnicity_encoded"], data["approval_encoded"])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-squared statistic: {chi2}")
    print(f"p-value: {p}")
    print(f"Degrees of freedom: {dof}")
    print(f"Expected frequencies:\n{expected}")


# chi_squared()

# train a random forest classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(trainX, trainY)
# predY = clf.predict(testX)
# print(accuracy_score(testY, predY))

# # save the model
# with open("model.pkl", "wb") as f:
#     pickle.dump(clf, f)

# # load the model
# with open("model.pkl", "rb") as f:
#     clf = pickle.load(f)
