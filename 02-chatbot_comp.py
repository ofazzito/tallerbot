from openai import OpenAI
import os

API_KEY = os.environ.get("API_KEY_OPENAI")

client = OpenAI(api_key = API_KEY)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
    {
      "role": "system",
      "content": "Eres un útil asistente experto en traducciones de idiomas."
    },
    {
      "role": "user",
      "content": "Traduce el siguiente texto en español al ingles: 'Hola, ¿cómo estás?'"
    },
    {
      "role": "assistant",
      "content": "'Hi, how are you?'"
    },
    {
      "role": "user",
      "content": "Ahora como se dice \"¡Buenos Días!\""
    }
  ],
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
) 

#print(response)

print(response.choices[0].message.content)