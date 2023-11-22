import webbrowser as web #abrir url de las imagenes y buscador
from openai import OpenAI 
import gradio as gr #interface
import logging as log #para llevar registros 

import pyttsx3 #convertir texto a voz_old
import pygame #reproducir mp3

import threading #para dividir la ejecucion de programas en hilos
import re, string
import os

#configuracion de registros
log.basicConfig(level=log.INFO)

# Inicializar pygame
#pygame.init()

# Configura tu clave de API de OpenAI
API_KEY = os.environ.get("API_KEY_OPENAI")

client = OpenAI(api_key = API_KEY)

# Función para obtener autocompletado
def bot_completion(prompt):
    response = client.completions.create(
        model="text-davinci-003",
        prompt=prompt, 
        max_tokens=100, #cantiad maxima de tokens en la respuesta
        n=1, #cantidad de respuestas
        stop=None, # caracter para marcar el fin de la respuesta
        temperature=0.5 # grado de creatividad 0-2 cuanto mas chico menos random       
        )
    
    return response.choices[0].text

#Funcion para el chat
def chat_completion(prompt, context): 
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
        ]
    )    
    texto = response.choices[0].message.content
    
    return texto

def reproducir_mp3(archivo):
    def reproducir():
        if not pygame.mixer.get_init():
            pygame.mixer.init() #  y el mezclador de audio una vez
        try:
            pygame.mixer.music.load(archivo)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except pygame.error as e:
            print(f"No se pudo reproducir el archivo: {e}")
        finally:
            pygame.mixer.music.stop()
            pygame.mixer.quit()

    # Creamos un hilo para reproducir el audio
    t = threading.Thread(target=reproducir)
    t.start()

def hablar(texto): #funcion para repoducir en audio el texto            
    log.info("Dando respuesta con voz...") #configuracion de voz assitente
    engine = pyttsx3.init() # configuración de voz del asistente
    engine.say(texto) # Se genera la voz a partir de un texto
    engine.runAndWait()   # Se reproduce la voz
    
def hablar_TTS(texto):
    response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input= texto
    )

    response.stream_to_file("speech.mp3")
    reproducir_mp3("speech.mp3")
        

def buscar_en_google(consulta):
    url = f"https://www.google.com/search?q={consulta}"
    web.open_new_tab(url)
    return url    
    

def procesar_chat(mensaje):
    resp = chat_completion(mensaje, "Eres un chatbot, muestrate proactivo y participativo")
    #resp = chat_completion(mensaje, "Eres un chatbot, mustrate con personalidad divertida, ironica e irritable")  
    return resp

def procesar_traduccion(prompt): 
    split = prompt.split(maxsplit=2)  
    #(Traduce) 
    # al + idioma + texto a traducir
    idioma, texto = split[1], split[2]
    log.info("Idioma: " + idioma)
    log.info("Frase: " + texto) 
    traduccion = chat_completion(texto, f"Eres un experto traductor, así que traduce esto al {idioma}")            
    return traduccion   

def procesar_dibujo(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt = prompt,
        n = 1, # cantidad de imagenes puede ir de 1 a 10
        size = "1024x1024" #tamaño
        )
    
    url = response.data[0].url
    web.open(url) # abre imagen en una pestaña
    return  url

def procesar_entrada(audio,voz):  
    #print(audio) 
    log.info("Procesando audio ...")
    audio_file = open(audio, "rb")
    transcript = client.audio.transcriptions.create(
        model ="whisper-1",
        file = audio_file
        )
   
    instruc, prompt = re.sub('[%s]' % re.escape(string.punctuation), '', transcript.text.split(maxsplit=1)[0]), transcript.text.split(maxsplit=1)[1]
    
    log.info("Transcripción: " + transcript.text)
    log.info("Instrucción: " + instruc)
    log.info("Prompt: " + prompt)
    
    # diccionario que mapea instrucciones a funciones
    acciones = {
        "Dime": [procesar_chat, True], # orden : [funcion a llamar, muestra texto si/no]
        "Traduce": [procesar_traduccion,True],
        "Dibuja": [procesar_dibujo, False],
        "Busca": [buscar_en_google, False]
    }
    
    if instruc.capitalize() in acciones:
        funcion = acciones[instruc.capitalize()][0]
        respuesta = funcion(prompt)
        if voz:
            if not acciones[instruc.capitalize()][1]:
                respuesta = "Abriendo URL"  
            # Creamos un hilo para ejecutar la función hablar()
            #t = threading.Thread(target=hablar, args=(respuesta,))
            t = threading.Thread(target=hablar_TTS, args=(respuesta,))
            t.start()         
        return respuesta
    else:
        resp = bot_completion(transcript.text)
        if voz:
            # Creamos un hilo para ejecutar la función hablar()
            #t = threading.Thread(target=hablar, args=(respuesta,))
            t = threading.Thread(target=hablar_TTS, args=(resp,))
            t.start()     
        return resp 

             
ui = gr.Interface(fn=procesar_entrada, 
                  inputs=[gr.Audio(sources="microphone",type="filepath"),
                          gr.Checkbox(label="activar voz")],
                  outputs="text",
                  title="Asistente de Prueba Multi Modelo",
                  description= '''Este asistente reconoce los comandos DIME, TRADUCE, DIBUJA, y BUSCA. Entendera 
                  cualquier frase que inicie con ellos. O simplemente completara la frase. Graba la orden y envia.
                  Si quieres que ademas responda por voz activa la casilla correspondiente''')
ui.launch()

