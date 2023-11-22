from openai import OpenAI
import os
import gradio as gr

# Configura tu clave de API de OpenAI
API_KEY = os.environ.get("API_KEY_OPENAI")

client = OpenAI(api_key = API_KEY)

#set contexto
messages = [{"role": "system",
             "content": "Eres un chatbot, muestrate proactivo y participativo. Da respuestas sencillas y de no mas de 4 oraciones"}]

def gpt35(pregunta,historial):
    messages.append({"role": "user", "content": pregunta})
    respuesta = client.chat.completions.create(
        model ="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,
    )
    
    texto = respuesta.choices[0].message.content
    return texto

iface = gr.ChatInterface(
    fn=gpt35,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Hazme una pregunta", container=False, scale=7),
    title="Chatbot de OpenAI",
    description="Escribe una pregunta y obtén una respuesta generada por ChatGPT."
    
)

if __name__ == "__main__":
    iface.launch()