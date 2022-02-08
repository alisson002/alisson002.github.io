# Lista de programação da dsiciplina Processamento Digital de Imagens (DCA0445)
Alisson Sousa Moreira - 20200149004 - alisson-mu@hotmail.com

[Exercício 7.2 - homomórfico](https://alisson002.github.io/#Exercício-72---homom%C3%B3rfico)

[Exercício 8.3 - cannypoints](https://alisson002.github.io/#Exercício-83---cannypoints)

[Exercício 9.2 - kmeans](https://alisson002.github.io/#Exercício-92----kmeans)

[Unidade 3 - Detecção de Faces](https://alisson002.github.io/#unidade-3----detector-de-faces)

## Exercício 7.2 - homomórfico

O processo de filtragem homomórfica é baseado nos princípios de iluminância e refletância. A iluminância representa variações espaciais lentas (frequências baixas), a refletância representa variações espaciais rápidas (frequências altas). Tomamos a transformada de Fourier da imagem e aplicamos um filtro sequencial (versão modificada do filtro gaussiano) que atenua as frequências baixas e mantém as frequências altas (filtro passa-altas) e fazemos a transformada de Fourier inversa, desta forma, melhorando a iluminância da imagem.

```python
import cv2
import numpy as np
from math import exp, sqrt
gh, gl, c, d0 = 0, 0, 0, 0
g, cv, dv = 0,0,0

def aplicaFiltro():
    global gh, gl, c, d0, complex
    du = np.zeros(complex.shape, dtype=np.float32)
    for u in range(dft_M):
        for v in range(dft_N):
            du[u,v] = sqrt((u-dft_M/2.0)*(u-dft_M/2.0)+(v-dft_N/2.0)*(v-dft_N/2.0))

    du2 = cv2.multiply(du,du) / (d0*d0)
    re = np.exp(- c * du2)
    H = (gh - gl) * (1 - re) + gl
   

    filtered = cv2.mulSpectrums(complex,H,0)
    

    filtered = np.fft.ifftshift(filtered)
    filtered = cv2.idft(filtered)
    filtered = cv2.magnitude(filtered[:,:,0], filtered[:,:,1])

    cv2.normalize(filtered,filtered,0, 1, cv2.NORM_MINMAX)
    filtered = np.exp(filtered)
    cv2.normalize(filtered, filtered,0, 1, cv2.NORM_MINMAX)

    cv2.imshow("homomorfico", filtered)

def setgl(g):
    global gl
    gl = g/10.0
    if gl > gh:
        gl = gh-1
        gl = g / 10.0
    aplicaFiltro()

def setgh(g):
    global gh
    gh = g/10.0
    if 1 > gh:
        gh = 1
        gh = g / 10.0
    if gl > gh:
        gh = gl + 1
        gh = g / 10.0
    aplicaFiltro()

def setc(cv):
    global c
    if cv == 0:
        cv = 1
    c = cv/1000.0
    aplicaFiltro()

def setd0(dv):
    global d0
    d0 = dv/10.0
    if d0 == 0:
        d0 = 1
        d0 = dv / 10.0
    aplicaFiltro()


image = cv2.imread("C:/Users/Alisson Moreira/Desktop/alisson002.github.io/dft.jpg", 0)
cv2.imshow("original", image)
cv2.imwrite("C:/Users/Alisson Moreira/Desktop/alisson002.github.io/dftcinza.jpg", image)
image = np.float32(image)
height, width = image.shape

dft_M = cv2.getOptimalDFTSize(height)
dft_N = cv2.getOptimalDFTSize(width)
padded = cv2.copyMakeBorder(image, 0, dft_M-height,0,dft_N-width, cv2.BORDER_CONSTANT, 0) + 1
padded = np.log(padded)
complex = cv2.dft(padded,flags=cv2.DFT_COMPLEX_OUTPUT)
complex = np.fft.fftshift(complex)
img_back = 20*np.log(cv2.magnitude(complex[:,:,0],complex[:,:,1]))
cv2.imshow("fft", np.uint8(img_back))
cv2.imwrite("C:/Users/Alisson Moreira/Desktop/alisson002.github.io/dftfft.jpg", np.uint8(img_back))
cv2.imshow("homomorfico", image)


trackbarName = "GL "
cv2.createTrackbar(trackbarName,"homomorfico",g,100,setgl)
trackbarName = "GH "
cv2.createTrackbar(trackbarName,"homomorfico",g,100,setgh)
trackbarName = "C "
cv2.createTrackbar(trackbarName,"homomorfico",cv,100,setc)
trackbarName = "D0 "
cv2.createTrackbar(trackbarName,"homomorfico",dv,dft_M,setd0)


cv2.waitKey(0)
cv2.destroyAllWindows()
```

Neste trecho do código a imagem é carregada em tons de cinza e coloca um padding para chegar no tamanho otimo para a FFT. Soma 1 em toda a imagem para não ter log(0), e depois é aplicado logaritmo neperiano em toda a imagem.
```python
image = cv2.imread("C:/Users/Alisson Moreira/Desktop/alisson002.github.io/dft.jpg", 0)
cv2.imshow("original", image)
cv2.imwrite("C:/Users/Alisson Moreira/Desktop/alisson002.github.io/dftcinza.jpg", image)
image = np.float32(image)
height, width = image.shape

dft_M = cv2.getOptimalDFTSize(height)
dft_N = cv2.getOptimalDFTSize(width)
padded = cv2.copyMakeBorder(image, 0, dft_M-height,0,dft_N-width, cv2.BORDER_CONSTANT, 0) + 1
padded = np.log(padded)
complex = cv2.dft(padded,flags=cv2.DFT_COMPLEX_OUTPUT)
complex = np.fft.fftshift(complex)
```
Imagen original em tons de cinza:
![tons de cinza](https://github.com/alisson002/alisson002.github.io/blob/main/dftcinza.jpg?raw=true)

Imagem do espectro de frequência:
![fft](https://github.com/alisson002/alisson002.github.io/blob/main/dftfft.jpg?raw=true)

Aqui é utilizada uma mascara como filtro do espectro frenquêncial: 

```python
def aplicaFiltro():
    global gh, gl, c, d0, complex
    du = np.zeros(complex.shape, dtype=np.float32)
    for u in range(dft_M):
        for v in range(dft_N):
            du[u,v] = sqrt((u-dft_M/2.0)*(u-dft_M/2.0)+(v-dft_N/2.0)*(v-dft_N/2.0))

    du2 = cv2.multiply(du,du) / (d0*d0)
    re = np.exp(- c * du2)
    H = (gh - gl) * (1 - re) + gl
   

    filtered = cv2.mulSpectrums(complex,H,0)
    

    filtered = np.fft.ifftshift(filtered)
    filtered = cv2.idft(filtered)
    filtered = cv2.magnitude(filtered[:,:,0], filtered[:,:,1])

    cv2.normalize(filtered,filtered,0, 1, cv2.NORM_MINMAX)
    filtered = np.exp(filtered)
    cv2.normalize(filtered, filtered,0, 1, cv2.NORM_MINMAX)

    cv2.imshow("homomorfico", filtered)
```
Depois, é feita a troca de quadrantes novamente e aplicada a transformada inversa. É retornada a magnitude dessa matriz, aplicada uma exponencial e normalizada para poder imprimir a imagem.

Imagem filtrada:

![imagem filtrada](https://github.com/alisson002/alisson002.github.io/blob/main/dftfiltrada.jpeg?raw=true)


## Exercício 8.3 - cannypoints
O programa abaixo faz o pontilhismo em uma imagem RGB:
```python
import cv2
import numpy as np
from random import *
from copy import copy

step = 5
jitter = 3
raio = 2
kernel = np.ones([3,3],dtype=np.uint8)

img = cv2.imread("C:/Users/Alisson Moreira/Desktop/alisson002.github.io/cannypoints.jpg")
image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

height, width, depth = img.shape
xrange, yrange = np.arange(0, int(height / step)), np.arange(0, int(width / step))

for i in range(len(xrange)):
    xrange[i] = int(xrange[i] * step + step / 2)
for j in range(len(yrange)):
    yrange[j] = int(yrange[j] * step + step / 2)

points = copy(img)
points = cv2.blur(points,(5,5))
points = cv2.blur(points,(5,5))
points = cv2.blur(points,(5,5))

vector = np.arange(10,200,10)
shuffle(vector)

for i in vector:
    np.random.shuffle(xrange)

    canny = cv2.Canny(image,i,3*i)
    canny = cv2.dilate(canny,kernel,iterations=1)

    for j in xrange:
        np.random.shuffle(yrange)
        for k in yrange:
            if canny[j,k] == 255:
                x = j + randrange(-jitter, jitter)
                y = k + randrange(-jitter, jitter)
                pixel = img[j, k].tolist()
                bola = int(raio + (i)/190*randrange(2,5))
                cv2.circle(points, (y, x), bola, pixel, -1, cv2.LINE_AA)

cv2.imshow("original", img)
cv2.imshow("pontilhada", points)
cv2.imwrite("C:/Users/Alisson Moreira/Desktop/alisson002.github.io/cannypointspontilhada.jpg", points)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
Ele percorre cada pixel da imagem e desenha um círculo de raio 2 com a mesma cor e na posição correspondente, usando um passo de 5 (escolhe um pixel para cada 5, nos eixos x e y).

Neste trecho do código é criada uma copia da imagem original que é borrada, para ser usada como fundo da imagem de pontilhismo:
```python
points = copy(img)
points = cv2.blur(points,(5,5))
points = cv2.blur(points,(5,5))
points = cv2.blur(points,(5,5))
```

Valores de treshold criados e embaralhados:
```python
vector = np.arange(10,200,10)
shuffle(vector)
```

Aqui é aplicado um procedimento parecido ao do exemplo ```pontilhismo.cpp```, com a diferença de que o procedimento é repetido para cada valor de treshold, para o qual é criada a imagem de bordas de Canny, dilatada com um kernel [3x3] e se o valor do pixel for 255 (indicando que tem uma borda nesse ponto) é criado o ponto utilizando o valor de cor da imagem original:
```python
for i in vector:
    np.random.shuffle(xrange)

    canny = cv2.Canny(image,i,3*i)
    canny = cv2.dilate(canny,kernel,iterations=1)

    for j in xrange:
        np.random.shuffle(yrange)
        for k in yrange:
            if canny[j,k] == 255:
                x = j + randrange(-jitter, jitter)
                y = k + randrange(-jitter, jitter)
                pixel = img[j, k].tolist()
                bola = int(raio + (i)/190*randrange(2,5))
                cv2.circle(points, (y, x), bola, pixel, -1, cv2.LINE_AA)
```

Imagem original:

![floresOriginal](https://github.com/alisson002/alisson002.github.io/blob/main/cannypoints.jpg?raw=true)

Imagem pontilhada:

![floresPontilhada](https://github.com/alisson002/alisson002.github.io/blob/main/cannypointspontilhada.jpg?raw=true)


## Exercício 9.2 -  kmeans
K-means é um processo de quantização que visa classificar N observações em K clusters.

No processamento digital de imagens, cada observação corresponde a um pixel, e os clusters são a quantidade de cores que queremos. Podemos ordenar cada pixel a partir da aproximação com cada centróide (um centróide por cluster), então, pegamos a distância média das amostras em cada cluster, criando novas posições de centróides. É um processo iterativo, esse processo acontece até não termos mudanças mais significativas nas posições dos centróides, enfim, podemos atribuir uma cor para cada cluster.

```python
import random 
import numpy as np
import cv2

nclusters = 8
attempts = 1

img = cv2.imread("C:/Users/Alisson Moreira/Desktop/alisson002.github.io/kmeans.jpg")
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
samples = img.reshape((-1,3))

samples = np.float32(samples)

for i in range(1, 11):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.1*random.randint(1,100))
    x, labels, centers = cv2.kmeans(samples, nclusters, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    vals = centers[labels.flatten()]
    newimg = vals.reshape((img.shape))

    cv2.imshow('New Image', newimg)
    cv2.imwrite("C:/Users/Alisson Moreira/Desktop/alisson002.github.io/kmeansGif/kmeans.png", newimg)
    cv2.waitKey(0) 

cv2.waitKey(0)
cv2.destroyAllWindows()
```
O programa acima gera uma saida diferente para cada uma das iterações, sendo 10 no total.

No trecho abaixo é feita a leitura e conversão da imagem original:
```python
img = cv2.imread("C:/Users/Alisson Moreira/Desktop/alisson002.github.io/kmeans.jpg")
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
samples = img.reshape((-1,3))

samples = np.float32(samples)
```

Aqui é onde as imagens são geradas:
```python
for i in range(1, 11):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.1*random.randint(1,100))
    x, labels, centers = cv2.kmeans(samples, nclusters, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    vals = centers[labels.flatten()]
    newimg = vals.reshape((img.shape))

    cv2.imshow('New Image', newimg)
    cv2.imwrite("C:/Users/Alisson Moreira/Desktop/alisson002.github.io/kmeansGif/kmeans.png", newimg)
    cv2.waitKey(0)  
```
 Usando KMEANS_RANDOM_CENTERS ao inves de KMEANS_PP_CENTERS cada posição inicial do Cluster é randomizada, gerando saídas diferentes a cada vez, para 10 rodadas, temos 10 saídas diferentes, conforme mostrado abaixo.

 Imagem original:
![wargraymonOriginal](https://github.com/alisson002/alisson002.github.io/blob/main/kmeans.jpg?raw=true)

Gif das imagens geradas:

![wargraymonGif](https://github.com/alisson002/alisson002.github.io/blob/main/kmeansGif/kmeans.gif?raw=true)


## Unidade 3 -  Detector de Faces
Este algoritmo tem como objetivo detectar e contar o número de faces em imagens e videos. O código deve colocar uma caixa (bounding box) no rostos detectados e contá-los, através de suas coordenadas.

Bibliotecas.
```python
import cv2
import dlib
```

Captura de vídeo com a webcam e captura de coordenadas.
```python
video = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
```

Neste trecho do código os frames do vídeo serão capturados continuamente e um iterador será iniciado. Cada vez que as coordenadas de um rosto forem capturadas o iterador será incrementado em 1, para que cada rosto seja exibido com seu respectivo número na caixa.
```python
while True:
    
    #captura frame a frame
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    #contador para contar o numero de rostos
    i = 0
    for face in faces:
 
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        #incrementa o interador cada vez que obtenho as coordenadas de um rosto
        i = i+1
        
        #adiciona o número do rosto a sua respectiva caixa
        cv2.putText(frame, 'face num'+str(i), (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(face, i)
 
    cv2.imshow('frame', frame)
    
    #termina o loop apertando a tecla 'e'
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
 
```

Algoritmo completo da detecção de faces com a webcam.
```python
import cv2
import dlib
 
 
video = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
 
 
while True:
 
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
 
    i = 0
    for face in faces:
 
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
 
        i = i+1
        
        cv2.putText(frame, 'face num'+str(i), (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(face, i)
 
    cv2.imshow('frame', frame)
 
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
 
 
video.release()
cv2.destroyAllWindows()
```

Algoritmo completo da detecção de faces em imagens.
```python
import cv2
import dlib
 
 
img = cv2.imread('C:/Users/Alisson Moreira/Desktop/alisson002.github.io/PDI unidade3/img5.jpeg')
 
detector = dlib.get_frontal_face_detector()
 
 
while True:
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
 
    i = 0
    for face in faces:
 
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)
 
        i = i+1
        
        cv2.putText(img, 'face num'+str(i), (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(face, i)
 
    cv2.imshow('imagem', img)
 
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
 
 
img.release()
cv2.destroyAllWindows()
```

![img01Original](https://github.com/alisson002/alisson002.github.io/blob/main/PDI%20unidade3/img1.jpg?raw=true)
![img01box](https://github.com/alisson002/alisson002.github.io/blob/main/PDI%20unidade3/img11.png?raw=true)

![img02Original](https://github.com/alisson002/alisson002.github.io/blob/main/PDI%20unidade3/img2.jpg?raw=true)
![img02box](https://github.com/alisson002/alisson002.github.io/blob/main/PDI%20unidade3/img22.png?raw=true)

![img03Original](https://github.com/alisson002/alisson002.github.io/blob/main/PDI%20unidade3/img3.jpg?raw=true)
![img03box](https://github.com/alisson002/alisson002.github.io/blob/main/PDI%20unidade3/img33.png?raw=true)

![img04Original](https://github.com/alisson002/alisson002.github.io/blob/main/PDI%20unidade3/i5.jpeg?raw=true)
![img04box](https://github.com/alisson002/alisson002.github.io/blob/main/PDI%20unidade3/img44.png?raw=true)

![img05Original](https://github.com/alisson002/alisson002.github.io/blob/main/PDI%20unidade3/img5.jpeg?raw=true)
![img05box](https://github.com/alisson002/alisson002.github.io/blob/main/PDI%20unidade3/img55.png?raw=true)

https://youtu.be/_ee5DCQmWaM