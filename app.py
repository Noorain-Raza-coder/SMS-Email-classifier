import streamlit as st
import pickle 
import nltk 
# from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import streamlit.components.v1 as components
nltk.download('punkt')

# creating classes
# vectorizer = TfidfVectorizer()
ps = PorterStemmer()


# importing model
model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))


# function that take user text i.e sms or email and return cleaned stemmed text
def TextCleaner(string):
    string = string.lower()      # 1)
    tokenized_Str = nltk.word_tokenize(string)  # 2)
    cleaned_text = []
    for item in tokenized_Str :
        if item.isalnum():
            if item not in stopwords.words('english'):
                cleaned_text.append(item)
    stemmed_text = []
    for word in cleaned_text:
        stemmed_text.append(ps.stem(word))
        
    return " ".join(stemmed_text)

# getting text from UI
st.markdown("""
<style>iframe {background-color: skyblue;}</style>
""", unsafe_allow_html=True)
components.html("<h1><div class='row' style='text-align:center'><font color='white'>---------------------: Welcome To :------------------- SMS / Email Spam classifier</font></div></h1>")
# st.markdown('''
#     :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
#     :gray[pretty] :rainbow[colors].''')
st.write(":red[Creator]: :orange[Noorain] :green[Raza]")
st.divider()

input_text = st.text_area("Enter Your email/SMS Below:")



if st.button(':red[Predict]',use_container_width=True):
    stemmed_text = TextCleaner(input_text)
    X = vectorizer.transform([stemmed_text])
    prediction = model.predict(X)

    if prediction[0] == 0:
        output = 'Not Spam'
    else:
        output = 'Spam'
    st.write(output)


