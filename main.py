import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig, GPT2LMHeadModel, GPT2Tokenizer
import openai
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

OPENAI_API_KEY = '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'

# Set up OpenAI API
openai.api_key = OPENAI_API_KEY

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# General preparations
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=BertConfig.from_pretrained('bert-base-uncased', from_tf=False))
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return lemmatized_tokens

# Information extraction
def extract_information(text):
    pattern_symptoms = r'\b(symptom|sign)'
    doc = nlp(text)
    symptoms = []
    for sentence in doc.sents:
        if re.search(pattern_symptoms, sentence.text, re.IGNORECASE):
            symptoms.extend([ent.text for ent in sentence.ents if ent.label_ == 'SYM'])
    return symptoms

# Text classification
def text_classification(documents):
    categories = ['disease', 'symptom', 'treatment']
    labels = [0, 1, 2]
    texts = [text for _, text, _ in documents]
    labels = [label for _, _, label in documents]
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_vectorized, y_train)
    y_pred = svm_classifier.predict(X_test_vectorized)

    return y_test, y_pred

# Text clustering
def text_clustering(texts):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(vectors)
    clusters = kmeans.predict(vectors)

    return clusters

# Language understanding model
def language_understanding(sentence):
    inputs = tokenizer.encode_plus(sentence, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.logits.mean(dim=1)  # Access the logits attribute instead of last_hidden_state
    return embeddings


# Text generation
def generate_text(prompt):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=100,
            top_p=1.0,
            n=1,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        text = response.choices[0].text.strip()
        return text
    except Exception as e:
        return f'OpenAI Generation Error: {str(e)}'

# Medical text title
title = "The Coronavirus: Molecular Mechanisms and Epidemiological Implications"

# Medical research text
text = '''
The coronavirus family, comprising zoonotic pathogens, gained global attention with the advent of severe acute respiratory syndrome coronavirus (SARS-CoV) in 2002, the Middle East respiratory syndrome coronavirus (MERS-CoV) in 2012, and the ongoing coronavirus disease 2019 (COVID-19) pandemic, caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). This article attempts to explore the molecular mechanisms that enable these coronaviruses to infect human hosts and the epidemiological implications thereof.
...
'''

# Tokenization and text preprocessing
processed_tokens = preprocess_text(text)

# Frequency distribution
freq_dist = FreqDist(processed_tokens)
print("Frequency Distribution:")
for word, frequency in freq_dist.most_common(10):
    print(word, ":", frequency)

# Information extraction
symptoms = extract_information(text)

# Display symptoms
print("\nSymptoms:")
for symptom in symptoms:
    print(symptom)

# Text classification
documents = [
    ('The Role of Vitamin D in Immune System Health', 'Vitamin D, also known as the "sunshine vitamin," plays a crucial role in various physiological processes and has been extensively studied for its impact on immune system health. This review aims to delve into the scientific evidence surrounding the role of vitamin D in supporting and regulating the immune system, highlighting its importance in maintaining overall health and preventing immune-related disorders.', 'treatment'),
    ('Common Symptoms of Influenza', 'Influenza, commonly known as the flu, is a contagious respiratory illness caused by influenza viruses. It presents with a range of symptoms, including fever, cough, sore throat, body aches, fatigue, and nasal congestion. Recognizing these common symptoms can help in early detection and prompt management of influenza.', 'symptom'),
    ('Understanding and Managing Diabetes Mellitus', 'Diabetes Mellitus is a chronic metabolic disorder characterized by high blood sugar levels. It affects millions of people worldwide and requires careful management to prevent complications. Understanding the causes, symptoms, and treatment options for diabetes is crucial for effective disease management and improved quality of life.', 'disease'),
    ('The Benefits of Regular Exercise for Cardiovascular Health', 'Regular exercise plays a vital role in maintaining cardiovascular health. It helps improve heart function, strengthen muscles, lower blood pressure, and reduce the risk of heart disease. By incorporating regular physical activity into daily routines, individuals can enjoy the numerous benefits that exercise provides for their cardiovascular well-being.', 'treatment'),
    ('Recognizing the Early Signs of Alzheimer\'s Disease', 'Alzheimer\'s disease is a progressive neurological disorder that primarily affects memory and cognitive function. Early recognition of the signs and symptoms, such as memory loss, confusion, and difficulty performing familiar tasks, is crucial for timely diagnosis and intervention. Understanding the early warning signs of Alzheimer\'s disease can lead to better management and improved quality of life for affected individuals and their families.', 'symptom'),
    ('Latest Advances in Cancer Treatment', 'Cancer treatment has undergone significant advancements in recent years, offering new hope for patients. From targeted therapies and immunotherapy to precision medicine and innovative surgical techniques, the field of oncology continues to evolve. Staying informed about the latest advances in cancer treatment can empower patients and healthcare professionals alike in making informed decisions regarding diagnosis and treatment options.', 'treatment'),
    ('Managing Chronic Pain: Strategies and Medications', 'Chronic pain is a complex and challenging condition that can significantly impact an individual\'s quality of life. Effective management of chronic pain involves a multimodal approach that combines various strategies, including medications, physical therapy, lifestyle modifications, and psychological interventions. By adopting a comprehensive pain management plan, individuals with chronic pain can experience improved well-being and functional abilities.', 'treatment'),
    ('Identifying Risk Factors for Heart Disease', 'Heart disease is a leading cause of mortality worldwide, and identifying risk factors is essential for prevention and early intervention. Common risk factors for heart disease include hypertension, high cholesterol, smoking, diabetes, obesity, and a sedentary lifestyle. Recognizing and addressing these risk factors can significantly reduce the chances of developing heart disease and promote cardiovascular health.', 'disease'),
]

# Text classification
y_test, y_pred = text_classification(documents)
classification_result = classification_report(y_test, y_pred, zero_division=0)

# Display classification report
print("\nClassification Report:")
print(classification_result)

# Text clustering
texts = [text for _, text, _ in documents]
clusters = text_clustering(texts)

# Display clustering results
print("\nClusters:")
for i, cluster in enumerate(clusters):
    print("Document:", texts[i])
    print("Cluster:", cluster)
    print()

# Language understanding
sentence = "The coronavirus family causes respiratory illnesses."
embeddings = language_understanding(sentence)

# Display sentence embedding
print("\nSentence Embedding:")
print(embeddings)

# Text generation
prompt = "Based on the patient's symptoms and medical history, the recommended treatment is"
generated_text = generate_text(prompt)
print("\nGenerated Text:")
print(generated_text)
