import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import fitz
data = {}
for i, category in enumerate(os.listdir('Data\\CVs')):

    for j, cv in enumerate(os.listdir(f'Data\\CVs\\{category}')):

        cv_id = cv.split('.')[0]
        cv = fitz.open(f'Data\\CVs\\{category}\\{cv}')
        data[cv_id] = ['\n'.join([page.get_text('text') for page in cv]), category]
        print(f'{i}.{j} Done.', end='\r')
resume_sections = ['summary', 'highlights', 'experience', 'education', 'certifications', 'skills', 'accomplishments']
resumes = {}

for resume_id, (resume, resume_category) in data.items():
    
    resumes[resume_id] = {}
    resumes[resume_id]['category'] = resume_category
    last_section = None
    
    for row in resume.lower().split('\n'):

        if row in resume_sections:
            
            last_section = row
            resumes[resume_id][row] = ""
        
        elif last_section:
            resumes[resume_id][last_section] += row + "\n"
data = pd.DataFrame(resumes).T
data = data.fillna("NO_VALUE")
job_description = ["Looking for a Python developer with experience in Machine Learning"]

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(data.skills)
vectors2 = vectorizer.transform(job_description)
pdf_name = data.iloc[(cosine_similarity(vectors2, vectors)).argmax()].category + "\\" + data.iloc[(cosine_similarity(vectors2, vectors)).argmax()].name
pdf_name
os.startfile(f"C:\\Users\\Hhjoo\\Desktop\\Projects\\Resume\\Data\\CVs\\{pdf_name}.pdf")