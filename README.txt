Controles

R: Guarda Un punto
W: Mueve la nave hacia delante
S: Mueve la nave hacia atras
A: Rotar la nave hacia la izquierda
D: Rotar la nave hacia la derecha
1: Mueve la nave por los puntos guardados
2: Detiene inmediateamente la nave
3: Borra la curva de puntos guardados
V: Dibuja la curva de puntos guardados (tuve un problema de pipeline con este boton lo que generaba aberraciones graficas, por lo que lo deje comentado en el codigo)
P: la nave comienza a girar como si se hubiera descontrolado a forma de pirueta finalizando en su posicion inicial
Mouse posicionado en la pantalla: Mueve la nave en el eje Y

Funcionamiento del programa

Cuando el usuario guarda 4 puntos de inmediato se crea una curva de bezier y se almacena, luego si el usuario vuelve a guardar 3 puntos
se crea otra curva de bezier y se almacena, y asi sucesivamente, cuando el usuario presiona la tecla 1 la nave se mueve por los puntos guardados
y si el usuario desea detener el movimiento puede apretar el boton 2, sin embargo al ocurrir 3 veces el loop de recorrido por las curvas se Detiene
de igual manera, cuando el usuario presiona la tecla 3 se borran todos los puntos guardados y curvas de bezier, y si el usuario desea ver la curva
que tiene en el momento puede presionar la tecla V y se dibujara la curva de bezier que se encuentra en el momento.


