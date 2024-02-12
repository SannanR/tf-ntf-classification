from io import StringIO
from Bio import SeqIO
import pandas as pd
import streamlit as st
import numpy as np
import math
import joblib
import base64
import pickle  # Import Pickle for model loading
from sklearn.preprocessing import StandardScaler

def seqToMat(seq):
    encoder = ['X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    len_seq = len(seq)
    n = int(math.ceil(math.sqrt(len_seq)))
    seqMat = [[0 for x in range(n)] for y in range(n)]
    seqiter = 0
    for i in range(n):
        for j in range(n):
            if seqiter < len_seq:
                try:
                    aa = int(encoder.index(seq[seqiter]))
                except ValueError:
                    exit(0)
                else:
                    seqMat[i][j] = aa
                seqiter += 1
    return seqMat


def frequencyVec(seq):
    encoder = ['X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
               'Y']
    fv = [0 for x in range(21)]
    i = 1
    for i in range(21):
        fv[i - 1] = seq.count(encoder[i])
    fv[20] = seq.count('X')
    return fv


def AAPIV(seq):
    encoder = ['X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
               'Y']
    apv = [0 for x in range(21)]
    i = 1
    sum = 0
    for i in range(21):
        j = 0
        for j in range(len(seq)):
            if seq[j] == encoder[i]:
                sum = sum + j + 1
        apv[i] = sum
        sum = 0
    return apv[1:] + apv[0:1]


def print2Dmat(mat):
    n = len(mat)
    i = 0
    strOut = ''
    for i in range(n):
        strOut = strOut + str(mat[i]) + '<br>'
    return strOut


def PRIM(seq):
    encoder = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
               'Y', 'X']
    prim = [[0 for x in range(21)] for y in range(21)]
    i = 0
    for i in range(21):
        aa1 = encoder[i]
        aa1index = -1
        for x in range(len(seq)):
            if seq[x] == aa1:
                aa1index = x + 1
                break
        if aa1index != -1:
            j = 0
            for j in range(21):
                if j != i:
                    aa2 = encoder[j]
                    aa2index = 0
                    for y in range(len(seq)):
                        if seq[y] == aa2:
                            aa2index = aa2index + ((y + 1) - aa1index)
                    prim[i][j] = int(aa2index)
    return prim


def rawMoments(mat, order):
    n = len(mat)
    rawM = []
    sum = 0
    i = 0
    for i in range(order + 1):
        j = 0
        for j in range(order + 1):
            if i + j <= order:
                p = 0
                for p in range(n):
                    q = 0
                    for q in range(n):
                        sum = sum + (((p + 1) ** i) * ((q + 1) ** j) * int(mat[p][q]))
                rawM.append(sum)
                sum = 0
    return rawM


def centralMoments(mat, order, xbar, ybar):
    n = len(mat)
    centM = []
    sum = 0
    i = 0
    for i in range(order + 1):
        j = 0
        for j in range(order + 1):
            if i + j <= order:
                p = 0
                for p in range(n):
                    q = 0
                    for q in range(n):
                        sum = sum + ((((p + 1) - xbar) ** i) * (((q + 1) - ybar) ** j) * mat[p][q])
                centM.append(sum)
                sum = 0
    return centM


def hahnMoments(mat, order):
    N = len(mat)
    hahnM = []
    i = 0
    for i in range(order + 1):
        j = 0
        for j in range(order + 1):
            if i + j <= order:
                answer = hahnMoment(i, j, N, mat)
                hahnM.append(answer)
    return hahnM


def hahnMoment(m, n, N, mat):
    value = 0.0
    x = 0
    for x in range(N):
        y = 0
        for y in range(N):
            value = value + (
                    mat[x][y] * (hahnProcessor(x, m, N)) * (hahnProcessor(x, n, N)))
    return value


def hahnProcessor(x, n, N):
    return hahnPol(x, n, N) * math.sqrt(roho(x, n, N))


def hahnPol(x, n, N):
    answer = 0.0
    ans1 = pochHammer(N - 1.0, n) * pochHammer(N - 1.0, n)
    ans2 = 0.0
    k = 0
    for k in range(n + 1):
        ans2 = ans2 + math.pow(-1.0, k) * ((pochHammer(-n, k) * pochHammer(-x, k) *
                                            pochHammer(2 * N - n - 1.0, k)))
    answer = ans1 + ans2
    return answer


def roho(x, n, N):
    return gamma(n + 1.0) * gamma(n + 1.0) * pochHammer((n + 1.0), N)


def gamma(x):
    return math.exp(logGamma(x))


def logGamma(x):
    temp = (x - 0.5) * math.log(x + 4.5) - (x + 4.5)
    ser = 101.19539853003
    return temp + math.log(ser * math.sqrt(2 * math.pi))


def pochHammer(a, k):
    answer = 1.0
    i = 0
    for i in range(k):
        answer = answer * (a + i)
    return answer
# (rest of your feature extraction functions)

def processAllStrings(fname):
    seqs = []
    allFVs = []
    with open(fname, 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            seqs.append(currentPlace)
    allowed_chars = set('ACDEFGHIKLMNPQRSTVWXY')
    i = 0
    for seq in seqs:
        print(str(i) + ': ' + seq)
        if seq != '':
            if set(seq).issubset(allowed_chars):
                allFVs.append(calcFV(seq))
                i = i + 1
            else:
                print('Invalid Sequence\n' + str(i))
                i = i + 1
    return allFVs

def calcFV(seq):
    fv = [0 for x in range(153)]  # Change 153 to 152
    fvIter = 0
    myMat = seqToMat(seq)
    myRawMoments = rawMoments(myMat, 3)
    for ele in myRawMoments:
        fv[fvIter] = ele
        fvIter = fvIter + 1
    xbar = myRawMoments[4]
    ybar = myRawMoments[1]
    myCentralMoments = centralMoments(myMat, 3, xbar, ybar)
    for ele in myCentralMoments:
        fv[fvIter] = ele
        fvIter = fvIter + 1
    myHahnMoments = hahnMoments(myMat, 3)
    for ele in myHahnMoments:
        fv[fvIter] = ele
        fvIter = fvIter + 1
    myFrequencyVec = frequencyVec(seq)
    for ele in myFrequencyVec:
        fv[fvIter] = ele
        fvIter = fvIter + 1
    myPRIM = PRIM(seq)
    myPRIMRawMoments = rawMoments(myPRIM, 3)
    xbar2 = myPRIMRawMoments[4]
    ybar2 = myPRIMRawMoments[1]
    myPRIMCentralMoments = centralMoments(myPRIM, 3, xbar2, ybar2)
    for ele in myPRIMRawMoments:
        fv[fvIter] = ele
        fvIter = fvIter + 1
    for ele in myPRIMCentralMoments:
        fv[fvIter] = ele
        fvIter = fvIter + 1
    myPRIMHahnMoments = hahnMoments(myPRIM, 3)
    for ele in myPRIMHahnMoments:
        fv[fvIter] = ele
        fvIter = fvIter + 1
    myAAPIV = AAPIV(seq)
    for ele in myAAPIV:
        fv[fvIter] = ele
        fvIter = fvIter + 1
    myRPRIM = PRIM(seq[::-1])
    myRPRIMRawMoments = rawMoments(myRPRIM, 3)
    xbar3 = myRPRIMRawMoments[4]
    ybar3 = myRPRIMRawMoments[1]
    myRPRIMCentralMoments = centralMoments(myRPRIM, 3, xbar3, ybar3)
    for ele in myRPRIMRawMoments:
        fv[fvIter] = ele
        fvIter = fvIter + 1
    for ele in myRPRIMCentralMoments:
        fv[fvIter] = ele
        fvIter = fvIter + 1
    myRPRIMHahnMoments = hahnMoments(myRPRIM, 3)
    for ele in myRPRIMHahnMoments:
        fv[fvIter] = ele
        fvIter = fvIter + 1
    myRAAPIV = AAPIV(seq[::-1])
    for ele in myRAAPIV:
        fv[fvIter] = ele
        fvIter = fvIter + 1
    return fv

def processAllStrings(fname):
    seqs = []
    allFVs = []
    with open(fname, 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            seqs.append(currentPlace)
    allowed_chars = set('ACDEFGHIKLMNPQRSTVWXY')
    i = 0
    for seq in seqs:
        print(str(i) + ': ' + seq)
        if seq != '':
            if set(seq).issubset(allowed_chars):
                allFVs.append(calcFV(seq))
                i = i + 1
            else:
                print('Invalid Sequence\n' + str(i))
                i = i + 1
    return allFVs

# Load the XGBoost model using pickle
def load_xgboost_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to preprocess input data for prediction
def preprocess_input_data(input_data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(input_data)
    return scaled_data

# Function to make predictions using the loaded model
def make_predictions(model, input_data):
    predictions = model.predict(input_data)
    return predictions

# Function to process the uploaded .fasta file, extract features, and make predictions
def process_and_predict(fasta_file, model):
    all_feature_vectors = []
    protein_names = []

    # Extract features for each sequence
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(seq_record.seq)
        protein_name = seq_record.id  # Extracting protein heading name
        protein_names.append(protein_name)

        feature_vector = calcFV(seq)  # Assuming calcFV is defined in your code
        all_feature_vectors.append(feature_vector)

    # Preprocess the feature vectors
    input_data = np.array(all_feature_vectors)

    # Exclude the last column (class labels)
    input_data = input_data[:, :-1]

    # Scale the data using StandardScaler
    scaled_data = preprocess_input_data(input_data)

    # Make predictions using the XGBoost model
    predictions = make_predictions(model, scaled_data)

    # Map predictions to "Non-TF" and "TF"
    class_mapping = {0: "Non-TF", 1: "TF"}
    mapped_predictions = [class_mapping[prediction] for prediction in predictions]

    # Combine protein names with predictions
    result = list(zip(protein_names, mapped_predictions))

    # Display the results in Streamlit
    for protein_name, prediction in result:
        st.write(f"Protein: {protein_name}, Label: {prediction}")


def main():
    st.title("Protein Prediction Web App") 
    file_ = open("./WebPic.gif", "rb")
    contents = file_.read()

    # Sidebar for user input# Sidebar for user input
    st.sidebar.subheader("Input Sequence(s) (FASTA FORMAT ONLY)")
    fasta_string = st.sidebar.text_area("Sequence Input", height=200)

    # Example button
    if st.button('Example'):
        example_sequences = [
            ">sp|O95476|CNEP1_HUMAN ...",  # Example sequence 1
            ">sp|Q99653|CHP1_HUMAN ...",  # Example sequence 2
            ">sp|P32929|CGL_HUMAN ...",  # Example sequence 3
        ]
        st.code("\n".join(example_sequences), language="markdown")

    # Submit button
    if st.sidebar.button("SUBMIT"):
        if fasta_string == "":
            st.info("Please input the sequence first.")
        else:
            fasta_io = StringIO(fasta_string)
            # Load the XGBoost model
            model_path = 'xgboost_model.joblib'  # Replace with the correct path to your model
            xgboost_model = load_xgboost_model(model_path)
            process_and_predict(fasta_io, xgboost_model)

if __name__ == "__main__":
    main()