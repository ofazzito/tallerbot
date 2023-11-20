from openai import OpenAI
import os
import gradio as gr

# Configura tu clave de API de OpenAI
API_KEY = os.environ.get("API_KEY_OPENAI")

client = OpenAI(api_key = API_KEY)

def generar_respuesta(pregunta,historia):
    respuesta = client.completions.create(
        model="text-davinci-003",
        prompt=pregunta,
        max_tokens=100
        
    )
    print (pregunta,historia)
    return respuesta.choices[0].text.strip()



iface = gr.ChatInterface(
    fn=generar_respuesta,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Hazme una pregunta", container=False, scale=7),  
    title="Chatbot de OpenAI",
    description="Escribe una pregunta y obtén una respuesta generada por ChatGPT."
    
)

if __name__ == "__main__":
    iface.launch()