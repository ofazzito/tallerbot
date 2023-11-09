import openai
import os

#from config import API_KEY

API_KEY = os.environ.get("API_KEY_OPENAI")
openai.api_key = API_KEY


def generar_respuesta(pregunta):
    respuesta = openai.completions.create(
        model="text-davinci-003",
        prompt=pregunta, 
        max_tokens= 60,
        temperature=0.5
    )
    return respuesta.choices[0].text.strip()

def main(): 
    print("Bienvenido al Chatbot de OpenAI")
    
    while True:
        pregunta = input("Escribe tu pregunta (o 'salir' para finalizar):" )
        
        if pregunta.lower() == 'salir':
            print("¡Hasta luego!")
            break
        
        repuesta = generar_respuesta(pregunta)
        print("Respuesta:", repuesta)
        
if __name__ == "__main__":
    main()