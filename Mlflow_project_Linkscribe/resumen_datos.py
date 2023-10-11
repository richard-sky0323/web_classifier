from bs4 import BeautifulSoup
import requests

def extraer_titulo_descripcion(self, link):
        response = requests.get(link)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            titulo = soup.title.string if soup.title else 'No se encontro título'
            descricion_tag = soup.find('meta', attrs={'name': 'description'})
            descripcion = descricion_tag['content'] if descricion_tag else 'No se encontro descripcion'
            return titulo, descripcion
        else:
            return "No es posible acceder a la pagina web", "No es posible acceder a la pagina web"