import pandas as pd

## Load in dataset -- Change file path
data = pd.read_csv("C:/Users/anebe/VSCode Projects/CSDS_313/cc-approval/clean_dataset.csv")

## Change gender col name to make more sense
data.rename(columns={"Gender":"Male"}, inplace=True)

## Create dummy variables for ethnicity -- White is the "default," as in all changes due to ethnicity are relative to being white
dummies = data['Ethnicity'].str.get_dummies()
data = pd.concat([data.drop(columns='Ethnicity'), dummies[dummies.columns[:-1]]], axis=1)

## Repeat for citizenship -- default is by birth
dummies = data['Citizen'].str.get_dummies()
data = pd.concat([data.drop(columns='Citizen'), dummies[dummies.columns[2:]]], axis=1)

## Condense Industry categories - default is other
"""
Utilities: Transport, Energy, Utilities
InfoSys: CommunicationServices, InformationTechnology
Consumer: ConsumerStaples, ConsumerDiscretionary
Finance: Financials, RealEstate
Industry: Industrials, Materials
Other: Education, Healthcare, Research
"""
data["Utilities"] = data['Industry'].isin(["Transport", "Energy", "Utilities"]).astype(int)
data["InfoSys"] = data['Industry'].isin(["CommunicationServices", "InformationTechnology"]).astype(int)
data["Consumer"] = data['Industry'].isin(["ConsumerStaples", "ConsumerDiscretionary"]).astype(int)
data["Finance"] = data['Industry'].isin(["Financials", "RealEstate"]).astype(int)
data["Industrial"] = data['Industry'].isin(["Industrials", "Materials"]).astype(int)
data.drop(columns="Industry")

print(data.head)
