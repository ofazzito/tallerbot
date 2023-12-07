from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory


import gradio as gr
import param
import os
import logging

chain_type = "stuff"

# Configura tu clave de API de OpenAI
API_KEY = os.environ.get("API_KEY_OPENAI")

# memory = ConversationSummaryBufferMemory(
#     llm=OpenAI(api_key=API_KEY), 
#     input_key="question", 
#     memory_key="resumen",
#     max_token_limit=1000,
# )

def load_files(files): 
    if isinstance(files, list):
        documents = []
        # load documents
        for fn in files:
            if os.path.isfile(fn):
                if fn.endswith('.pdf'):
                    loader = PyPDFLoader(fn)
                    documents.extend( loader.load())
        if len(documents) > 0:
            # split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=150)
            docs = text_splitter.split_documents(documents)
            return docs
    return []

#funcion para cargar base de datos
def get_chroma_db(embeddings, documents, path,):

    if chat.recreate_db:
        logging.info("RECREANDO CHROMA DB")
        db = Chroma.from_documents(
            documents=documents, embedding=embeddings, persist_directory=path
            )
        db.persist()
        return db
    else:
        logging.info("CARGANDO CHROMA EXISTENTE")
        return Chroma(persist_directory=path, embedding_function=embeddings)

def load_db(files):
    
    docs = load_files(files)
    # define embedding
    embeddings = OpenAIEmbeddings(api_key=API_KEY)
    # create vector database from data
    persist_path = 'DB/croma_const'
    db = get_chroma_db(embeddings, docs, persist_path)    
    return db
   
def create_qa(db,chain_type, k):
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    
    # Construir el prompt
    #Resumen conversacion:{resumen}
    template = """Eres un asistente cordial y proactivo, que ayuda al usario con sus preguntas. \
    Si el usario te saluda responde el saludo muy amablemente y luego pregunta en que lo puedes ayudar \
    Si logras entender la pregunta del usurio entonces: \    
    use las siguientes piezas de contexto para responder la pregunta al final. \
    Si no sabe la respuesta, simplemente diga que no sabe, no intente inventar una respuesta. \
    Utilice 6 oraciones como máximo. Mantenga la respuesta lo más concisa posible. \
    Por favor dar las respuesta en español \
    Cuando respondas una pregunta usando el contexto puedes cada tanto decir \
    "¡Gracias por preguntar!" al final de la respuesta.
    

    Contexto:{context}
    
    Pregunta: {question}
    
    Respuesta:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question" ],template=template,)  
    chain_type_kwargs = {"prompt":QA_CHAIN_PROMPT}
    
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.25,api_key=API_KEY), 
        chain_type=chain_type,
        retriever=retriever, 
        chain_type_kwargs=chain_type_kwargs,
        #memory=memory
    )
    
    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.25,api_key=API_KEY),    
    #     memory=memory,
    #     retriever=retriever,
    #     chain_type=chain_type,
    #     condense_question_prompt = chain_type_kwargs
    # )
    return qa 

def process_files(files, recreate_chroma_db):
    chat.recreate_db = recreate_chroma_db
    if files is None and recreate_chroma_db:
        return "¡Debes indicar un archivo antes de procesarlo!"
    if chat.recreate_db:
        for file in files:
            # Si el archivo está cargado, guarda el path y realiza alguna acción
            path = os.path.join(os.getcwd(), file)
            chat.loaded_file.append(path)
            logging.info(f"Archivo cargado: {path}")
    db = load_db(files)
    qa = create_qa(db,"stuff", 4)
    chat.qa = qa
    
    return "Archivos procesados!!! Ahora puede hacer sus consultas"

def process_chat(question,history):
    
    if chat.loaded_file is [] and chat.recreate_db :
        return "¡Debes cargar un archivo antes de enviar consulta!"
    db = load_db(chat.loaded_file)
    qa = create_qa(db,"stuff", 4)
    chat.qa = qa
    response = chat.convchain(question)
    logging.info("Respuesta: " + response)
    
    return response    

class chatbot(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    
    def __init__(self,  **params):
        super(chatbot, self).__init__( **params)
        self.loaded_file = []
        self.qa = None
        self.recreate_db = False
    
    
    def convchain(self, query):
        result = self.qa.run(query)
        # self.chat_history.extend([(query, result["answer"])])
        # self.db_query = result["generated_question"]
        # self.db_response = result["source_documents"]
        # self.answer = result['answer']
        return result
 

ifiles = gr.Interface(
    fn=process_files,
    inputs=[
        gr.Files(label="Seleccione los archivos PDFs", file_count="multiple", ),
        gr.Checkbox(label="Recrear base de datos", value=False)
    ],
    outputs=gr.components.Textbox(),
    description="Cargue un archivo PDF, procese y realice consultas para obtener respuestas basadas en el contenido del archivo.",
)


ichat = gr.Interface(
    fn=process_files,
    inputs=[
        gr.components.Textbox(label="Ingrese una Consulta de Texto"),
    ],
    outputs=gr.components.Textbox(),
    description="Realice consultas para obtener respuestas basadas en el contenido del archivo.",
)

itchat = gr.ChatInterface(
    fn=process_chat,
    description="Realice consultas para obtener respuestas basadas en el contenido del archivo.",
    chatbot= gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="Haz tu consulta aqui", container=False, scale=7),
)

iface =  gr.TabbedInterface(
    interface_list=[itchat, ifiles],
    tab_names=["Chat", "Carga archivo"],
    title="Interfaz de Consulta de Archivos PDF",    
)

if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)
    chat = chatbot()
    iface.launch()