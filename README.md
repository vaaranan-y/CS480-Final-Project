# CS 480 - Final Project

### Course: CS 480

### Professor: Yaoliang Yu

### Student: Vaaranan Yogalingam

## Project Title

**Random Forest Regression Model Feasibility for
Plant Data Analysis**

## Overview

This project repository contains all accompanying files that have also been submitted on LEARN. Instructions on how to navigate this repository (and in turn, the project submitted on LEARN) can be found below.

## Table of Contents

- [Getting Started](#gettingStarted)
- [Overview](#overview)
- [Data](#data)
- [Report](#report)
- [Code](#code)
- [Submissions](#submissions)

## Getting Started

```bash
git clone https://github.com/vaaranan-y/CS480-Final-Project.git
cd CS480-Final-Project/code
pip install -r requirements.txt
```

## Data

This is an exact copy and paste of the data files provided for this project. No new files were added, and no modifications were made.

## Report

This folder contains the rendered LaTeX file for the report, along with all the required files to render the report in Overleaf. The rendered report is a pdf file called "CS480_Report_Vaaranan_Yogalingam.pdf"

## Code

This folder contains the code for both models. Each script has 1 of 2 names:

1. model_no_image.py
2. model_with_image.py
   Either of these models can be ran as so:

```bash
python3 [MODEL_NAME].py [x]
```

Where [x] is the number of estimators you would like the model to use (to see the relevance of this value, please refer to the project report). This value is mandatory, and a good test value would be 100. Here is an example command:

```bash
python3 model_no_image.py 200
```

## Submissions

This folder contains the prediction CSV files that were submitted to Kaggle and whose scores were used in the final analysis of the report (i.e. R^2 value versus number of estimators used). Each file has the following file name:

```bash
20901584_Yogalingam_[included or blank]_[number of iterations].csv
```

Here is an example file name:

```bash
20901584_Yogalingam_included_300.csv
```
