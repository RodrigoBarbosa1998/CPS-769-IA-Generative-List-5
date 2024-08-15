import os
import pdfplumber
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

class ArticleRetriever:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.articles = []
        self.embeddings = []
        self.texts = []
        self._load_articles()
        self._generate_embeddings()

    def _load_articles(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        for filename in os.listdir(data_dir):
            if filename.endswith('.pdf'):
                file_path = os.path.join(data_dir, filename)
                self.articles.append(self._read_pdf(file_path))

    def _read_pdf(self, file_path):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text

    def _generate_embeddings(self):
        for article in self.articles:
            chunks = self._split_text_into_chunks(article)
            for chunk in chunks:
                self.texts.append(chunk)
                embedding = self._get_embedding(chunk)
                self.embeddings.append(embedding)
        self.embeddings = np.array(self.embeddings)

    def _split_text_into_chunks(self, text, max_tokens=500):
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += 1
            if current_length >= max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _get_embedding(self, text):
        response = self.client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def query(self, question):
        question_embedding = self._get_embedding(question)
        similarities = cosine_similarity([question_embedding], self.embeddings)[0]
        most_similar_index = np.argmax(similarities)
        most_similar_chunk = self.texts[most_similar_index]

        # Ajuste para a chamada do chat
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um especialista em inteligência artifical e generativa."},
                {"role": "user", "content": f"Question: {question}\nContext: {most_similar_chunk}"}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    api_key = "********************************************************"  
    retriever = ArticleRetriever(api_key)

    # Perguntas predefinidas
    predefined_questions = [
        "Qual é o assunto principal do artigo?",
        "Quem são os autores do artigo?",
        "Quais são os principais resultados do artigo?",
        "Qual é a metodologia utilizada no artigo?",
        "Quais são as conclusões do artigo?",
        "Como o artigo contribui para o campo de estudo?",
        "Quais são as limitações do estudo apresentado?",
        "Há referências importantes citadas no artigo?",
    ]

    # Loop para perguntas predefinidas
    for question in predefined_questions:
        answer = retriever.query(question)
        print(f"Pergunta: {question}")
        print(f"Resposta: {answer}")
        print()

    # Loop para perguntas livres
    while True:
        question = input("Digite sua pergunta sobre o artigo (ou 'sair' para encerrar): ")
        if question.lower() == 'sair':
            break
        answer = retriever.query(question)
        print("Resposta:", answer)

