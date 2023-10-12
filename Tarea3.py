import os.path
import sys
import numpy as np
import pyglet
from OpenGL.GL import *
import grafica.basic_shapes as bs
import grafica.easy_shaders as es
import grafica.transformations as tr
import grafica.scene_graph as sg
import grafica.lighting_shaders as ls
from grafica.assets_path import getAssetPath
from grafica.gpu_shape import createGPUShape
from pyglet.window import Window, key, mouse
assets= {
    "nave": getAssetPath("nave.obj"),
    "pochita_obj": getAssetPath("pochita3.obj"),
    "navei": getAssetPath("navei.obj"),
    "metal": getAssetPath("metaltex.jpg"),
    "muro" : getAssetPath("muro.obj"),
    "muralla": getAssetPath("muralla.obj"),
    "cilindro": getAssetPath("cilindro.obj"),
    "sombra": getAssetPath("sombra.obj")
}
class Controller(pyglet.window.Window):

    def __init__(self, width, height, title="Naves"):
        super().__init__(width, height, title)
        self.total_time = 0.0
        self.fillPolygon = True
        self.showAxis = True
        self.repeats = 0
        self.viewPos = np.array([30,30,30])
        self.at = np.array([0,0,0])
        self.camUp = np.array([0, 10, 0])
        self.distance = 20

        self.rotYder=False
        self.rotYizq=False
        self.xr=False
        self.xa=False
        self.rotzder=False
        self.rotzizq=False
        self.movement = True
        self.step = 0
        self.N= 100
        self.BezierCurve= []
        self.P=[]
        self.R=[]
        self.firstpos = []
        self.bucle=0
        self.curves=0
        self.Dibujo=False
        self.curvaNode=0
        self.lines={}
        self.pirueta=False
        self.curvapirueta=0
        self.Ppirueta=[]
        self.N2=20
        self.movement2=True
        self.L=0
    def update(self, nodo, sombra):
        position= sg.findPosition(nodo, "naveM")
        #position1= sg.findPosition(nodo, "naveH")
        #position2= sg.findPosition(nodo, "naveH_2")
        findpos= nodo.transform
        if self.xr == True:
            nodo.transform = tr.matmul([nodo.transform, tr.translate(-0.1,0,0)])
            
        if self.xa == True:
            nodo.transform = tr.matmul([nodo.transform, tr.translate(0.1,0,0)])
        if self.rotzder==True: 
            nodo.transform = tr.matmul([nodo.transform, tr.rotationZ(0.05)])
        if self.rotzizq==True: 
            nodo.transform = tr.matmul([nodo.transform, tr.rotationZ(-0.05)])
        
        if self.rotYder == True:
            nodo.transform = tr.matmul([nodo.transform, tr.rotationY(0.05)])
        if self.rotYizq == True: 
            nodo.transform = tr.matmul([nodo.transform, tr.rotationY(-0.05)])

        sombra.childs[0].transform = tr.translate(position[0][0],0,position[2][0])
        #sombra.childs[1].transform = tr.translate(position1[0][0],0,position1[2][0])
        #sombra.childs[0].transform = tr.translate(position2[0][0],0,position2[2][0])
        
        self.viewPos = np.array([findpos[0][3]+30,findpos[1][3]+30,findpos[2][3]+30])
        self.at = np.array([findpos[0][3],findpos[1][3],findpos[2][3]])
        
        
        return nodo


def readFaceVertex(faceDescription):

    aux = faceDescription.split('/')

    assert len(aux[0]), "Vertex index has not been defined."

    faceVertex = [int(aux[0]), None, None]

    assert len(aux) == 3, "Only faces where its vertices require 3 indices are defined."

    if len(aux[1]) != 0:
        faceVertex[1] = int(aux[1])

    if len(aux[2]) != 0:
        faceVertex[2] = int(aux[2])

    return faceVertex

def readOBJ(filename, color):

    vertices = []
    normals = []
    textCoords= []
    faces = []

    with open(filename, 'r') as file:
        for line in file.readlines():
            aux = line.strip().split(' ')
            
            if aux[0] == 'v':
                vertices += [[float(coord) for coord in aux[1:]]]

            elif aux[0] == 'vn':
                normals += [[float(coord) for coord in aux[1:]]]

            elif aux[0] == 'vt':
                assert len(aux[1:]) == 2, "Texture coordinates with different than 2 dimensions are not supported"
                textCoords += [[float(coord) for coord in aux[1:]]]

            elif aux[0] == 'f':
                N = len(aux)                
                faces += [[readFaceVertex(faceVertex) for faceVertex in aux[1:4]]]
                for i in range(3, N-1):
                    faces += [[readFaceVertex(faceVertex) for faceVertex in [aux[i], aux[i+1], aux[1]]]]

        vertexData = []
        indices = []
        index = 0

        # Per previous construction, each face is a triangle
        for face in faces:

            # Checking each of the triangle vertices
            for i in range(0,3):
                vertex = vertices[face[i][0]-1]
                normal = normals[face[i][2]-1]

                vertexData += [
                    vertex[0], vertex[1], vertex[2],
                    color[0], color[1], color[2],
                    normal[0], normal[1], normal[2]
                ]

            # Connecting the 3 vertices to create a triangle
            indices += [index, index + 1, index + 2]
            index += 3        

        return bs.Shape(vertexData, indices)
def read_OBJ2(filename):

    vertices = []
    normals = []
    tex_coords = []
    faces = []

    with open(filename, 'r') as file:
        for line in file.readlines():
            aux = line.strip().split(' ')

            if aux[0] == 'v':
                vertices += [[float(coord) for coord in aux[1:]]]

            elif aux[0] == 'vn':
                normals += [[float(coord) for coord in aux[1:]]]

            elif aux[0] == 'vt':
                assert len(aux[1:]) == 2, "Texture coordinates with different than 2 dimensions are not supported"
                tex_coords += [[float(coord) for coord in aux[1:]]]

            elif aux[0] == 'f':
                N = len(aux)
                faces += [[readFaceVertex(face_vertex) for face_vertex in aux[1:4]]]
                for i in range(3, N-1):
                    faces += [[readFaceVertex(face_vertex) for face_vertex in [aux[i], aux[i+1], aux[1]]]]

        vertex_data = []
        indices = []
        index = 0

        # Per previous construction, each face is a triangle
        for face in faces:

            # Checking each of the triangle vertices
            for i in range(0, 3):
                vertex = vertices[face[i][0]-1]
                texture = tex_coords[face[i][1]-1]
                normal = normals[face[i][2]-1]

                vertex_data += [
                    vertex[0], vertex[1], vertex[2],
                    texture[0], texture[1],
                    normal[0], normal[1], normal[2]
                ]

            # Connecting the 3 vertices to create a triangle
            indices += [index, index + 1, index + 2]
            index += 3

    return bs.Shape(vertex_data, indices)
def setPlot(texPipeline, axisPipeline, lightPipeline):
    projection = tr.ortho(-8, 8, -8, 8, 0.1, 100)

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "lightPosition"),0, 8, -15)
    
    glUniform1ui(glGetUniformLocation(lightPipeline.shaderProgram, "shininess"), 1000)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "constantAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "quadraticAttenuation"), 0.01)

def setView(texPipeline, axisPipeline, lightPipeline):
    view = tr.lookAt(
            controller.viewPos,
            controller.at,
            controller.camUp
        )

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "viewPosition"), controller.viewPos[0], controller.viewPos[1], controller.viewPos[2])
    
def createTexturedArc(d):
    vertices = [d, 0.0, 0.0, 0.0, 0.0,
                d+1.0, 0.0, 0.0, 1.0, 0.0]
    
    currentIndex1 = 0
    currentIndex2 = 1

    indices = []

    cont = 1
    cont2 = 1

    for angle in range(4, 185, 5):
        angle = np.radians(angle)
        rot = tr.rotationY(angle)
        p1 = rot.dot(np.array([[d],[0],[0],[1]]))
        p2 = rot.dot(np.array([[d+1],[0],[0],[1]]))

        p1 = np.squeeze(p1)
        p2 = np.squeeze(p2)
        
        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont/4])
        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont/4])
        
        indices.extend([currentIndex1, currentIndex2, currentIndex2+1])
        indices.extend([currentIndex2+1, currentIndex2+2, currentIndex1])

        if cont > 4:
            cont = 0


        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont/4])
        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont/4])

        currentIndex1 = currentIndex1 + 4
        currentIndex2 = currentIndex2 + 4
        cont2 = cont2 + 1
        cont = cont + 1

    return bs.Shape(vertices, indices)

def createTiledFloor(dim):
    vert = np.array([[-0.5,0.5,0.5,-0.5],[-0.5,-0.5,0.5,0.5],[0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0]], np.float32)
    rot = tr.rotationX(-np.pi/2)
    vert = rot.dot(vert)

    indices = [
         0, 1, 2,
         2, 3, 0]

    vertFinal = []
    indexFinal = []
    cont = 0

    for i in range(-dim,dim,1):
        for j in range(-dim,dim,1):
            tra = tr.translate(i,0.0,j)
            newVert = tra.dot(vert)

            v = newVert[:,0][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 1])
            v = newVert[:,1][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 1])
            v = newVert[:,2][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 0])
            v = newVert[:,3][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 0])
            
            ind = [elem + cont for elem in indices]
            indexFinal.extend(ind)
            cont = cont + 4

    return bs.Shape(vertFinal, indexFinal)

def createStaticScene(pipeline):

    sandBaseShape = createGPUShape(pipeline, createTiledFloor(50))
    sandBaseShape.texture = es.textureSimpleSetup(
        getAssetPath("fondo2.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)



    sandNode = sg.SceneGraphNode('sand')
    sandNode.transform = tr.translate(0.0,-0.1,0.0)
    sandNode.childs += [sandBaseShape]

    scene = sg.SceneGraphNode('system')
    scene.childs += [sandNode]
    
    return scene
class Shape:
    def __init__(self, vertices, indices):
        self.vertices = vertices
        self.indices = indices

    def __str__(self):
        return "vertices: " + str(self.vertices) + "\n"\
            "indices: " + str(self.indices)

    

def createLineColor(P1, P2, R, G, B):
    vertex = [
        P1[0], P1[1], P1[2], R, G, B,
        P2[0], P2[1], P2[2], R, G, B
    ]

    index = [0, 1]

    return Shape(vertex, index)
# Se asigna el ancho y alto de la ventana y se crea.
WIDTH, HEIGHT = 1000, 1000
width=WIDTH
height=HEIGHT
controller = Controller(width=WIDTH, height=HEIGHT)

# Se configura OpenGL
glClearColor(1.0, 1.0, 1.0, 1.0)
# Como trabajamos en 3D, necesitamos chequear cuáles objetos están en frente, y cuáles detrás.
glEnable(GL_DEPTH_TEST)

# Se configura el pipeline y se le dice a OpenGL que utilice ese shader
axisPipeline = es.SimpleModelViewProjectionShaderProgram()
texPipeline = es.SimpleTextureModelViewProjectionShaderProgram()
lightPipeline = ls.SimpleGouraudShaderProgram()
glUseProgram(axisPipeline.shaderProgram)

def crearNave(pipeline):
    naveM = createGPUShape(pipeline, readOBJ(assets["navei"], (1,0,0)))
    #naveH = createGPUShape(pipeline, readOBJ(assets["navei"], (0,1,0)))
    #naveH_2 = createGPUShape(pipeline, readOBJ(assets["navei"], (0,0,1)))


    naveNode = sg.SceneGraphNode("naveM")
    naveNode.transform= tr.uniformScale(0.5)
    naveNode.childs += [naveM]
    #navehija1 = sg.SceneGraphNode("naveH")
    #navehija1.childs += [naveH]
    #navehija1.transform= tr.matmul([tr.translate(-3,0,-3), tr.uniformScale(0.5)])
    #navehija2 = sg.SceneGraphNode("naveH_2")
    #navehija2.childs += [naveH_2]
    #navehija2.transform= tr.matmul([tr.translate(-3,0,3), tr.uniformScale(0.5)])

    flota=sg.SceneGraphNode("flota")

    flota.childs +=[naveNode]
    #flota.childs +=[navehija1]
    #flota.childs +=[navehija2]



    return flota
def crearSombras(pipeline):
    sombra1 = createGPUShape(pipeline, readOBJ(assets["sombra"], (0,0,0)))
    #sombra2 = createGPUShape(pipeline, readOBJ(assets["sombra"], (0,0,0)))
    #sombra3 = createGPUShape(pipeline, readOBJ(assets["sombra"], (0,0,0)))



    naveNodesombra = sg.SceneGraphNode("sombra1")
    naveNodesombra.childs +=[sombra1]
    naveNodesombra.transform= tr.uniformScale(0.1)
    #navehija1sombra = sg.SceneGraphNode("sombra2")
    #navehija1sombra.childs +=[sombra2]
    #navehija1sombra.transform= tr.matmul([tr.translate(-3,-0.1,-3), tr.uniformScale(0.3)])

    #navehija2sombra = sg.SceneGraphNode("sombra3")
    #navehija2sombra.childs +=[sombra3]
    #navehija2sombra.transform= tr.matmul([tr.translate(-3,-0.1,3), tr.uniformScale(0.3)])

    sombras=sg.SceneGraphNode("sombras")

    #sombras.childs +=[navehija2sombra]
    #sombras.childs +=[navehija1sombra]
    sombras.childs +=[naveNodesombra]

    return sombras

def crearMuro(pipeline, R, G, B):
    naveShape = createGPUShape(pipeline, readOBJ(assets["muro"], (R,G,B)))
    naveNode = sg.SceneGraphNode("muro")
    naveNode.childs += [naveShape]
    return naveNode
def crearMuralla(pipeline, R, G, B):
    naveShape = createGPUShape(pipeline, readOBJ(assets["muralla"], (R,G,B)))
    naveNode = sg.SceneGraphNode("muralla")
    naveNode.childs += [naveShape]
    return naveNode
def crearCurva(pipeline,M, R, G, B):
    lineas={}
    for i in range(len(M)-1):
        lineas[i]=createGPUShape(pipeline, createLineColor(M[i],M[i+1],R,G,B))
    return lineas
def crearCilindro(pipeline, R, G, B):
    naveShape = createGPUShape(pipeline, readOBJ(assets["cilindro"], (R,G,B)))
    naveNode = sg.SceneGraphNode("cilindro")
    naveNode.childs += [naveShape]
    return naveNode

    
naveNode = crearNave(lightPipeline)
naveNode.transform = tr.translate(1.0,0.5,1.0)
naveNode.transform = tr.uniformScale(0.1)

muroNode = crearMuro(lightPipeline, 0, 1, 1)
muroNode.transform = tr.scale(0.5,0.5,0.5)
muroNode.transform = tr.translate(7.0,0,-10.0)

muroNode2 = crearMuro(lightPipeline, 0, 1, 1)
muroNode2.transform = tr.scale(0.5,0.5,0.5)
muroNode2.transform = tr.translate(-30.0,0,-10.0)

cilindroNode = crearCilindro(lightPipeline, 0, 1, 1)
cilindroNode.transform = tr.scale(0.5,0.5,0.5)
cilindroNode.transform = tr.translate(0.0,0,-20.0)

murallaNode = crearMuralla(lightPipeline, 0, 1, 1)
murallaNode.transform = tr.scale(0.5,0.5,0.5)
murallaNode.transform = tr.translate(15.0,0,-25.0)

sombraNode = crearSombras(lightPipeline)
sombraNode.transform = tr.scale(0.5,0.5,0.5)
naveNode.transform = tr.translate(1.0,0.2,1.0)



scene = createStaticScene(texPipeline)
scene.transform = tr.translate(0.0,0.0,0.0)
scene.transform = tr.uniformScale(2)

# Creating shapes on GPU memory
cpuAxis = bs.createAxis(7)
gpuAxis = es.GPUShape().initBuffers()
axisPipeline.setupVAO(gpuAxis)
gpuAxis.fillBuffers(cpuAxis.vertices, cpuAxis.indices, GL_STATIC_DRAW)

def update(dt, window):
    window.total_time += dt

def generateT(t):
    return np.array([[1, t, t**2, t**3]]).T


def bezierMatrix(P0, P1, P2, P3):
    
    # Generate a matrix concatenating the columns
    G = np.concatenate((P0, P1, P2, P3), axis=1)
    # Bezier base matrix is a constant
    Mb = np.array([[1, -3, 3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])

    return np.matmul(G, Mb)

# M is the cubic curve matrix, N is the number of samples between 0 and 1
def evalCurve(M, N):
    # The parameter t should move between 0 and 1
    ts = np.linspace(0.0, 1.0, N)
    
    # The computed value in R3 for each sample will be stored here
    curve = np.ndarray(shape=(N, 3), dtype=float)
    
    for i in range(len(ts)):
        T = generateT(ts[i])
        curve[i, 0:3] = np.matmul(M, T).T
        
    return curve
def pirueta():
    #gira en circulos dependiendo de p0
    p0=(controller.Ppirueta[0])
    p1=(controller.Ppirueta[0])+np.array([[0],[0],[1]])
    p2=(controller.Ppirueta[0])+np.array([[1],[0],[1]])
    p3=(controller.Ppirueta[0])+np.array([[1],[0],[0]])
    curvae=bezierMatrix(p0,p1,p2,p3)
    beziercurve=evalCurve(curvae,controller.N2)
    p4=(controller.Ppirueta[0])+np.array([[1],[0],[0]])
    p5=(controller.Ppirueta[0])+np.array([[1],[0],[-1]])
    p6=(controller.Ppirueta[0])+np.array([[0],[0],[-1]])
    p7=(controller.Ppirueta[0])+np.array([[-1],[0],[-1]])
    curvae=bezierMatrix(p4,p5,p6,p7)
    beziercurve2=evalCurve(curvae,controller.N2)
    p8=(controller.Ppirueta[0])+np.array([[-1],[0],[-1]])
    p9=(controller.Ppirueta[0])+np.array([[-1],[0],[0]])
    p10=(controller.Ppirueta[0])+np.array([[-1],[0],[1]])
    p11=(controller.Ppirueta[0])+np.array([[0],[0],[1]])
    curvae=bezierMatrix(p8,p9,p10,p11)
    beziercurve3=evalCurve(curvae,controller.N2)
    p12=(controller.Ppirueta[0])+np.array([[0],[0],[1]])
    p13=(controller.Ppirueta[0])+np.array([[1],[0],[1]])
    p14=(controller.Ppirueta[0])+np.array([[1],[0],[0]])
    p15=(controller.Ppirueta[0])
    curvae=bezierMatrix(p12,p13,p14,p15)
    beziercurve4=evalCurve(curvae,controller.N2)
    curvae=np.concatenate((beziercurve,beziercurve2,beziercurve3,beziercurve4), axis=0)
    return curvae

@controller.event
def on_draw():
    controller.clear()

    # Si el controller está en modo fillPolygon, dibuja polígonos. Si no, líneas.
    if controller.fillPolygon:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    else:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    if controller.movement == False:
        if controller.step >= controller.N*controller.curves -1:
            controller.step = 0
            controller.bucle += 1

        controller.step += 1
    if controller.movement2 == False:
        if controller.step >= controller.N2*4 -1:
            controller.step = 0
            controller.bucle += 1
            if controller.bucle == 3:
                naveNode.transform = controller.L

        controller.step += 1
    if controller.bucle == 3:
        controller.step = 0
        controller.bucle = 0
        controller.firstpos = []
        controller.movement = True
        controller.movement2 = True

    # Using the same view and projection matrices in the whole application
    setView(texPipeline, axisPipeline, lightPipeline)
    setPlot(texPipeline, axisPipeline, lightPipeline)

    if controller.showAxis:
        glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "model"), 1, GL_TRUE,
                           tr.identity())
        axisPipeline.drawCall(gpuAxis, GL_LINES)
    #creamos una curva de bezier con los 2 puntos que se ingresan
    if len(controller.P)==4:
        curva=bezierMatrix(controller.P[0],controller.P[1],controller.P[2],controller.P[3])
        beziercurve=evalCurve(curva,controller.N)
        if controller.BezierCurve == []:
            controller.BezierCurve= beziercurve
            controller.curves += 1
            #controller.lines = crearCurva(lightPipeline, controller.BezierCurve, 0, 0, 1)
        else:
            controller.BezierCurve = np.concatenate((controller.BezierCurve,beziercurve), axis=0)
            controller.curves += 1
            #controller.lines = crearCurva(lightPipeline, controller.BezierCurve, 0, 0, 1)
        controller.P=[controller.P[3]]
    #if controller.Dibujo== True:
        #for i in range(len(controller.lines)-1):
            #glUseProgram(lightPipeline.shaderProgram)
            #axisPipeline.drawCall(controller.lines[i], GL_LINES)  

    # En cada iteración actualizamos la posición del cubo verde
    # Llamamos a la curva de Bezier definida
    if controller.pirueta==True:
        beziercurve2=pirueta()
        controller.curvapirueta=beziercurve2
    if controller.movement2==False:
        if controller.step < controller.N2*4-1:
            angle = np.arctan2(controller.curvapirueta[controller.step+1,0]-controller.curvapirueta[controller.step,0], controller.curvapirueta[controller.step+1,2]-controller.curvapirueta[controller.step,2])
        else:
            angle = np.arctan2(controller.curvapirueta[0,0]-controller.curvapirueta[controller.step,0],controller.curvapirueta[0,2]-controller.curvapirueta[controller.step,2])
        #angulo Z
        if controller.step < controller.N2*4-1:
            angle3 = np.arctan2(controller.curvapirueta[controller.step+1,1]-controller.curvapirueta[controller.step,1],controller.curvapirueta[controller.step,2]-controller.curvapirueta[controller.step+1,2])
        else:
            angle3 = np.arctan2(controller.curvapirueta[controller.step-1,1]-controller.curvapirueta[controller.step,1], controller.curvapirueta[controller.step-1,2]-controller.curvapirueta[controller.step,2])
        transformGreen = tr.matmul([
                tr.translate(controller.curvapirueta[controller.step, 0], controller.curvapirueta[controller.step, 1], controller.curvapirueta[controller.step, 2]),tr.rotationY(angle-np.pi/2),tr.rotationZ(angle3)])
        naveNode.transform = transformGreen



    if controller.movement == False:
        #angulo Y
        if controller.step < controller.N*controller.curves-1:
            angle = np.arctan2(controller.BezierCurve[controller.step+1,0]-controller.BezierCurve[controller.step,0], controller.BezierCurve[controller.step+1,2]-controller.BezierCurve[controller.step,2])
        else:
            angle = np.arctan2(controller.BezierCurve[0,0]-controller.BezierCurve[controller.step,0],controller.BezierCurve[0,2]-controller.BezierCurve[controller.step,2])
        #angulo Z
        if controller.step < controller.N*controller.curves-1:
            angle3 = np.arctan2(controller.BezierCurve[controller.step+1,1]-controller.BezierCurve[controller.step,1],controller.BezierCurve[controller.step,2]-controller.BezierCurve[controller.step+1,2])
        else:
            angle3 = np.arctan2(controller.BezierCurve[controller.step-1,1]-controller.BezierCurve[controller.step,1], controller.BezierCurve[controller.step-1,2]-controller.BezierCurve[controller.step,2])

        #transformacion
        if controller.step  == 1:
            naveNode.transform = controller.firstpos
        else:
            transformGreen = tr.matmul([
                tr.translate(controller.BezierCurve[controller.step, 0], controller.BezierCurve[controller.step, 1], controller.BezierCurve[controller.step, 2]),tr.rotationY(angle-np.pi/2),tr.rotationZ(angle3)])
            naveNode.transform = transformGreen





    glUseProgram(texPipeline.shaderProgram)
    sg.drawSceneGraphNode(scene, texPipeline, "model")

    # dibujar la nave
    glUseProgram(lightPipeline.shaderProgram)

    sg.drawSceneGraphNode(sombraNode, lightPipeline, "model")
    sg.drawSceneGraphNode(muroNode, lightPipeline, "model")
    sg.drawSceneGraphNode(muroNode2, lightPipeline, "model")
    sg.drawSceneGraphNode(cilindroNode, lightPipeline, "model")
    sg.drawSceneGraphNode(murallaNode, lightPipeline, "model")
    sg.drawSceneGraphNode(naveNode, lightPipeline, "model")


    controller.update(naveNode,sombraNode)

@controller.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.S:
        if controller.movement == True and controller.movement2== True:
            controller.xr=True
    if symbol == pyglet.window.key.W:
        if controller.movement == True and controller.movement2==True:
            controller.xa=True
    if symbol == pyglet.window.key.D:
        if controller.movement == True and controller.movement2== True:        
            controller.rotYizq=True
    if symbol == pyglet.window.key.A:
        if controller.movement == True and controller.movement2== True:
            controller.rotYder=True
    if symbol == pyglet.window.key.R:
        L= naveNode.transform
        rotx= np.arctan2(L[2][1],L[2][2])
        roty= np.arctan2(-L[2][0],np.sqrt(L[2][1]**2+L[2][2]**2))
        rotz= np.arctan2(L[1][0],L[0][0])
        pos = np.array([[L[0][3]],[L[1][3]],[L[2][3]]])

        controller.P.append(pos)
        rot= np.array([[rotx],[roty],[rotz]])
        rot = np.transpose([rot])
        controller.R.append(rot)
        if len(controller.P)==1 and controller.firstpos == []:
            controller.firstpos = L
    if symbol == pyglet.window.key._1:
        if controller.curves >=1:
            controller.movement = False
    if symbol == pyglet.window.key._2:
        controller.movement = True
        controller.step = 0
        controller.bucle = 0
    if symbol == pyglet.window.key._3:
        controller.BezierCurve = 0
        controller.curves = 0
    #if symbol == pyglet.window.key.V:
        #controller.Dibujo=True
    if symbol == pyglet.window.key.P:
        L= naveNode.transform
        pos = np.array([[L[0][3]],[L[1][3]],[L[2][3]]])
        controller.Ppirueta=[]
        controller.Ppirueta.append(pos)
        controller.pirueta=True
        controller.movement2= False
        controller.L=L
@controller.event
def on_key_release(symbol, modifiers):
    if symbol == pyglet.window.key.S:
        controller.xr=False
    if symbol == pyglet.window.key.W:
        controller.xa=False
    if symbol == pyglet.window.key.D:
        controller.rotYizq=False
    if symbol == pyglet.window.key.A:
        controller.rotYder=False
#movimiento con solo mover el mouse
@controller.event
def on_mouse_motion(x,y,dx,dy):
    if (y>700):
        controller.rotzder=True
    if (y<300):
        controller.rotzizq=True
    if (300<y<700):
        controller.rotzizq=False
        controller.rotzder=False

# Try to call this function 60 times per second

pyglet.clock.schedule(update, controller)
# Se ejecuta la aplicación
pyglet.app.run()
