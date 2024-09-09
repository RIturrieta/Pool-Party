""" Programa principal """

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.performance_monitor as pm
import grafica.scene_graph as sg
import grafica.text_renderer as tx
from grafica.assets_path import *
from math import *
import shaders as sh
from obj_reader import *
import sys
import json

# Parámetros de config.json

jason = str(sys.argv[1])
config = open(jason,)
data = json.load(config)

config.close()

R = data["Factor de friccion cinetica"]
C = data["Coeficiente de restitucion"]
MODE = data["Tecnica EDO (Euler|Euler modificado|Euler mejorado|Runge Kutta 4)"]
assert (MODE == "Euler") or (MODE == "Euler modificado") or (MODE == "Euler mejorado") or (MODE == "Runge Kutta 4")
HUD = data["HUD (On|Off)"]
assert (HUD == "On") or (HUD == "Off")
L = data["Mapa calor luz (On|Off)"]
assert (L == "On") or (L == "Off")

#####################################################################

class Controller:
    def __init__(self):
        self.camara = False

        self.action = True
        self.golpe = True

        # Parámetros de la cámara cercana al palo
        self.theta = np.pi
        self.eye = np.array([3.0, 0.0,-1.5])
        self.at = np.array([3.0, 0.5,-1.5])
        self.up = np.array([0.0, 0.0, 1.0])


    def on_key(self, window, key, scancode, action, mods):

        # Movimiento rápido
        if key == glfw.KEY_W:
            if (abs(controller.eye[0] + (controller.at[0] - controller.eye[0]) * 0.07) < 5.5) and (abs(controller.eye[1] + (controller.at[1] - controller.eye[1]) * 0.07) < 3.775):
                controller.eye += (controller.at - controller.eye) * 0.07
                controller.at += (controller.at - controller.eye) * 0.07
        elif key == glfw.KEY_S:
            if (abs(controller.eye[0] + (controller.at[0] - controller.eye[0]) * -0.07) < 5.5) and (abs(controller.eye[1] + (controller.at[1] - controller.eye[1]) * -0.07) < 3.775):
                controller.eye += (controller.at - controller.eye) * -0.07
                controller.at += (controller.at - controller.eye) * -0.07
        elif key == glfw.KEY_D:
                controller.theta -= np.pi*0.02
        elif key == glfw.KEY_A:
                controller.theta += np.pi*0.02

        # Movimiento preciso
        elif key == glfw.KEY_UP:
            if action == glfw.PRESS:    
                if (abs(controller.eye[0] + (controller.at[0] - controller.eye[0]) * 0.07) < 5.5) and (abs(controller.eye[1] + (controller.at[1] - controller.eye[1]) * 0.07) < 3.775):
                    controller.eye += (controller.at - controller.eye) * 0.02
                    controller.at += (controller.at - controller.eye) * 0.02
        elif key == glfw.KEY_DOWN:
            if action == glfw.PRESS:
                if (abs(controller.eye[0] + (controller.at[0] - controller.eye[0]) * -0.07) < 5.5) and (abs(controller.eye[1] + (controller.at[1] - controller.eye[1]) * -0.07) < 3.775):    
                    controller.eye += (controller.at - controller.eye) * -0.02
                    controller.at += (controller.at - controller.eye) * -0.02
        elif key == glfw.KEY_RIGHT:
            if action == glfw.PRESS:
                controller.theta -= np.pi*0.005
        elif key == glfw.KEY_LEFT:
            if action == glfw.PRESS:
                controller.theta += np.pi*0.005
        
        # Golpear
        if key == glfw.KEY_SPACE and self.golpe:
            if action == glfw.PRESS:
                if self.action:
                    self.golpe = False

        # Cambio de camara
        if key == glfw.KEY_1:
            if action == glfw.PRESS:
                self.camara = not self.camara

        # Cerrar la ventana
        if key == glfw.KEY_ESCAPE:
            if action == glfw.PRESS:
                glfw.set_window_should_close(window, True)


#####################################################################

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 1080
    height = 820
    title = "T3b - Pool"

    # Suavizado de bordes
    glfw.window_hint(glfw.SAMPLES, 8)
    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    controller = Controller()
    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, controller.on_key)

    # Enabling transparencies
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Setting up the clear screen color
    glClearColor(0.25, 0.25, 0.25, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    #####################################################################

    # Pelotas
    
    TexPipeline = sh.TexturePhongShaderProgram()

    blanca = createTextureGPUShapeR(readOBJT(getAssetPath('ball.obj')),TexPipeline, "sprites/sphere1.png")
    amarilla = createTextureGPUShapeR(readOBJT(getAssetPath('ball.obj')),TexPipeline, "sprites/sphere2.png")
    roja = createTextureGPUShapeR(readOBJT(getAssetPath('ball.obj')),TexPipeline, "sprites/sphere3.png")
    negra = createTextureGPUShapeR(readOBJT(getAssetPath('ball.obj')),TexPipeline, "sprites/sphere4.png")

    shadowPipeline = sh.SimpleModelViewProjectionShaderProgram()
    circle = bs.createColorCircle(20,0,0.15,0)
    gpucircle = es.GPUShape().initBuffers()
    shadowPipeline.setupVAO(gpucircle)
    gpucircle.fillBuffers(circle.vertices, circle.indices, GL_STATIC_DRAW)

    # Como son pocas pelotas, definimos las posiciones iniciales a mano
    posiciones = [[0.0,0.0],[-3.0,0.1],[-3.0,-0.1],[-3.0,-0.3],[-3.0,0.3],[-3.2,-0.2],[-3.2,0.0],
    [-3.2,0.4],[-3.2,0.2],[-2.8,0.2],[-3.2,-0.4],[-2.6,-0.1],[-2.6,0.1],[-2.4,0.0],[-2.8,-0.2],[-2.8,0.0]]
    
    ball = []

    balls = sg.SceneGraphNode("balls")
    balls.childs = []

    sombras = sg.SceneGraphNode("sombras")
    sombras.childs = []

    class pelota:
        def __init__(self,id):
            self.id = id
            self.pos = np.array(posiciones[id])
            self.radius = 0.1
            self.velocity = np.array([0.0,0.0])
            self.mass = 1.0

        def action(self, deltatime):
            if MODE == "Euler":
                self.actionE(deltatime)
            if MODE == "Euler modificado":
                self.actionEMo(deltatime)
            if MODE == "Euler mejorado":
                self.actionEMe(deltatime)
            if MODE == "Runge Kutta 4":
                self.actionRK4(deltatime)

        def actionE(self, deltatime):
            # Euler        
            self.velocity -= deltatime * self.velocity *(R*self.mass)
            self.pos += self.velocity * deltatime

        def actionEMo(self, deltatime):
            # Euler modificado
            vel = self.velocity - self.velocity *(deltatime/2) * (R*self.mass)
            self.velocity -= deltatime * vel
            self.pos += self.velocity * deltatime

        def actionEMe(self, deltatime):
            # Euler mejorado
            vel1 = self.velocity
            vel2 = self.velocity - self.velocity * deltatime * (R*self.mass)
            self.velocity -= deltatime * 0.5 * (vel1 + vel2)    
            self.pos += self.velocity * deltatime

        def actionRK4(self, deltatime):
            # Runge Kutta de 4 etapas
            k1 = self.velocity
            k2 = self.velocity - (deltatime/2) * (R*self.mass) * k1
            k3 = self.velocity - (deltatime/2) * (R*self.mass) * k2
            k4 = self.velocity - deltatime * (R*self.mass) * k3
            self.velocity -= (deltatime/6) * (k1 + 2*k2 + 2*k3 + k4)
            self.pos += self.velocity * deltatime

    # Función para crear pelotas, asignandoles el color y posicion correspondientes al Blackball
    def crearPelota(id):
        thisBall = sg.SceneGraphNode(str(id))
        thisBall.transform = tr.uniformScale(0.1)
        if id == 0:
            thisBall.childs = [blanca]
        elif id == 15:
            thisBall.childs = [negra]
        elif id%2 == 0:
            thisBall.childs = [amarilla]
        else:
            thisBall.childs = [roja]

        ballPos = sg.SceneGraphNode(str(id)+"c")
        ballPos.childs = [thisBall] 

        balls.childs += [ballPos]
        
        ballS = pelota(id)
        ball.append(ballS)

        # A cada pelota se le asigna su propia sombra
        circleNode = sg.SceneGraphNode(str(id))
        circleNode.transform = tr.uniformScale(0.25)
        circleNode.childs = [gpucircle]

        sombra1 = sg.SceneGraphNode(str(id)+"1")
        sombra1.childs = [circleNode]

        sombra2 = sg.SceneGraphNode(str(id)+"2")
        sombra2.childs = [sombra1]

        sombras.childs += [sombra2]

    # Se crean las pelotas
    for i in range(len(posiciones)):
        crearPelota(i)

    # Funciones para las colisiones de las pelotas
    def collide(circle1, circle2):
        """
        If there are a collision between the circles, it modifies the velocity of
        both circles in a way that preserves energy and momentum.
        """
        
        assert isinstance(circle1, pelota)
        assert isinstance(circle2, pelota)

        normal = circle2.pos - circle1.pos
        normal /= np.linalg.norm(normal)

        circle1MovingToNormal = np.dot(circle2.velocity, normal) > 0.0
        circle2MovingToNormal = np.dot(circle1.velocity, normal) < 0.0

        if not (circle1MovingToNormal and circle2MovingToNormal):
            # masas
            m1 = circle1.mass
            m2 = circle2.mass
            # impulso
            j = (1+C)*np.dot((circle2.velocity - circle1.velocity),normal)/(1/m1 + 1/m2)

            # this means that we applying energy and momentum conservation
            circle1.velocity += j*normal/m1
            circle2.velocity -= j*normal/m2

    # Funcion especial para la colision con el palo
    def hit(circle1):
        """
        If there are a collision between the circles, it modifies the velocity of
        both circles in a way that preserves energy and momentum.
        """
        circle2 = np.array([controller.at[0],controller.at[1]])

        assert isinstance(circle1, pelota)

        speed = 40*(circle1.pos-circle2)
        normal = circle1.pos - circle2
        normal /= np.linalg.norm(normal)

        m1 = circle1.mass
        m2 = 2
        
        # impulso
        j = (1+C)*np.dot((speed - circle1.velocity),normal)/(1/m1 + 1/m2)

        circle1.velocity += j*normal/m1

    # Verificar si dos pelotas están chocando
    def areColliding(circle1, circle2):
        assert isinstance(circle1, pelota)
        assert isinstance(circle2, pelota)

        difference = circle2.pos - circle1.pos
        distance = np.linalg.norm(difference)
        collisionDistance = circle2.radius + circle1.radius
        return distance < collisionDistance

    # Ajuste en la velocidad de las pelotas cuando chocan con el borde de la mesa, excepto cuando pasan por los hoyos
    def collideWithBorder(circle):
        # Right
        if circle.pos[0] + circle.radius > 4.3 and not((circle.pos[1] < circle.radius + 0.25 -2.575) or (-circle.radius - 0.25 + 2.575 < circle.pos[1])):
            circle.velocity[0] = -abs(circle.velocity[0])

        # Left
        if circle.pos[0] < -4.3 + circle.radius and not((circle.pos[1] < circle.radius + 0.25 -2.575) or (-circle.radius - 0.25 + 2.575 < circle.pos[1])):
            circle.velocity[0] = abs(circle.velocity[0])

        # Top
        if circle.pos[1] > 2.575 - circle.radius and not(((-circle.radius - 0.25 < circle.pos[0]) and (circle.pos[0] < circle.radius + 0.25)) or (circle.pos[0] < circle.radius + 0.25 -4.3) or (-circle.radius - 0.25 +4.3 < circle.pos[0])):
            circle.velocity[1] = -abs(circle.velocity[1])

        # Bottom
        if circle.pos[1] < -2.575 + circle.radius and not(((-circle.radius - 0.25 < circle.pos[0]) and (circle.pos[0] < circle.radius + 0.25)) or (circle.pos[0] < circle.radius + 0.25 -4.3) or (-circle.radius - 0.25 +4.3 < circle.pos[0])):
            circle.velocity[1] = abs(circle.velocity[1])

    ############################################

    # Función para crear la skybox/habitación

    def create_skybox(pipeline):
        shapeSky = bs.createTextureQuad(1,1)
        gpuSky = es.GPUShape().initBuffers()
        pipeline.setupVAO(gpuSky)
        gpuSky.fillBuffers(shapeSky.vertices, shapeSky.indices, GL_STATIC_DRAW)
        gpuSky.texture = es.textureSimpleSetup(
            getSpritePath("f4.png"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)        
        #################################################################################################
        shapeSecondSky = bs.createTextureQuad(1,1)
        gpuSecondSky = es.GPUShape().initBuffers()
        pipeline.setupVAO(gpuSecondSky)
        gpuSecondSky.fillBuffers(shapeSecondSky.vertices, shapeSecondSky.indices, GL_STATIC_DRAW)
        gpuSecondSky.texture = es.textureSimpleSetup(
            getSpritePath("f2.png"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
        #################################################################################################
        shapeThirdSky = bs.createTextureQuad(1,1)
        gpuThirdSky = es.GPUShape().initBuffers()
        pipeline.setupVAO(gpuThirdSky)
        gpuThirdSky.fillBuffers(shapeThirdSky.vertices, shapeThirdSky.indices, GL_STATIC_DRAW)
        gpuThirdSky.texture = es.textureSimpleSetup(
            getSpritePath("f3.png"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
        #################################################################################################
        shapeFourthSky = bs.createTextureQuad(1,1)
        gpuFourthSky = es.GPUShape().initBuffers()
        pipeline.setupVAO(gpuFourthSky)
        gpuFourthSky.fillBuffers(shapeFourthSky.vertices, shapeFourthSky.indices, GL_STATIC_DRAW)
        gpuFourthSky.texture = es.textureSimpleSetup(
            getSpritePath("f1.png"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
        #################################################################################################
        shapeFifthSky = bs.createTextureQuad(1,1)
        gpuFifthSky = es.GPUShape().initBuffers()
        pipeline.setupVAO(gpuFifthSky)
        gpuFifthSky.fillBuffers(shapeFifthSky.vertices, shapeFifthSky.indices, GL_STATIC_DRAW)
        gpuFifthSky.texture = es.textureSimpleSetup(
            getSpritePath("f6.png"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
        #################################################################################################
        d = 30

        skybox = sg.SceneGraphNode("skybox")
        skybox.transform = tr.matmul([tr.translate(-d, 0, -1.5), tr.rotationX(math.pi/2), tr.rotationY(math.pi/2), tr.uniformScale(d*2)])
        skybox.childs += [gpuSky]

        ##########################################################
        secondSky = sg.SceneGraphNode("secondSky")
        secondSky.transform = tr.matmul([tr.translate(d, 0, -1.5), tr.rotationX(math.pi/2), tr.rotationY(math.pi/-2), tr.uniformScale(d*2)])
        secondSky.childs += [gpuSecondSky]

        ##########################################################
        thirdSky = sg.SceneGraphNode("thirdSky")
        thirdSky.transform = tr.matmul([tr.translate(0, d, -1.5), tr.rotationX(math.pi/2), tr.uniformScale(d*2)])
        thirdSky.childs += [gpuThirdSky]

        ##########################################################
        fourthSky = sg.SceneGraphNode("fourthSky")
        fourthSky.transform = tr.matmul([tr.translate(0, -d, -1.5), tr.rotationX(math.pi/2), tr.rotationY(math.pi), tr.uniformScale(d*2)])
        fourthSky.childs += [gpuFourthSky]

        ##########################################################
        fifthSky = sg.SceneGraphNode("fifthSky")
        fifthSky.transform = tr.matmul([tr.translate(0, 0, -d-1.5), tr.uniformScale(d*2)])
        fifthSky.childs += [gpuFifthSky]

        ##########################################################
        newSkybox = sg.SceneGraphNode("secondSky")
        newSkybox.transform = tr.identity()
        newSkybox.childs += [skybox, secondSky, thirdSky, fourthSky, fifthSky]
        ############################################################
        return newSkybox
    
    ############################################

    # Mesa

    table = createTextureGPUShape(readOBJT(getAssetPath('pooltable.obj')),TexPipeline, "sprites/pooltable.png")
    tableNode = sg.SceneGraphNode("table")
    tableNode.transform = tr.matmul([tr.translate(0,0,-5),tr.rotationX(pi/2),tr.uniformScale(0.65)])
    tableNode.childs = [table]
    
    ############################################

    # Palo 

    stick = createTextureGPUShapeR(readOBJT(getAssetPath('poolstick.obj')),TexPipeline, "sprites/pstick.png")
    stickNode = sg.SceneGraphNode("stick")
    stickNode.childs = [stick]

    stickg = sg.SceneGraphNode("stickg")
    stickg.transform = tr.matmul([tr.translate(0,-0.25,0),tr.rotationY(pi),tr.rotationZ(0.075*pi),tr.uniformScale(0.65)])
    stickg.childs = [stickNode]
    
    stickt = sg.SceneGraphNode("stickt")
    stickt.childs = [stickg]
    
    
    ############################################

    # Habitación

    roomPipeline = es.SimpleTextureModelViewProjectionShaderProgram()

    skybox = create_skybox(roomPipeline)
    skyboxNode = sg.SceneGraphNode("skybox")
    skyboxNode.transform = tr.matmul([tr.translate(0,0,0),tr.uniformScale(100)])
    skyboxNode.childs = [skybox]
    

    ############################################

    # Referencias a nodos para aplicar transformaciones

    palo = sg.findNode(stickt, "stickt")
    tiro = sg.findNode(stickNode, "stick")

    ############################################

    # Camara fija

    feye = np.array([0.0,0.0,5.0])
    fat = np.array([0.0,0.0,0.0])
    fup = np.array([0.0,1.0,0.0])

    ######################################################
    
    # Texto del HUD

    textPipeline = tx.TextureTextRendererShaderProgram()
    
    # Creating texture with all characters
    textBitsTexture = tx.generateTextBitsTexture()
    # Moving texture to GPU memory
    gpuText3DTexture = tx.toOpenGLTexture(textBitsTexture)
            
    canpress = "Puede golpear?:"
    pressSize = 0.065
    pressShape = tx.textToShape(canpress, pressSize, pressSize)
    gpuPress = es.GPUShape().initBuffers()
    textPipeline.setupVAO(gpuPress)
    gpuPress.fillBuffers(pressShape.vertices, pressShape.indices, GL_STATIC_DRAW)
    gpuPress.texture = gpuText3DTexture
    pressTransform = tr.translate(-0.95, 0.9, 0)
    pressTransformd = tr.translate(-0.925, 0.915, 0)

    si1 = "si"
    si1Size = 0.065
    si1Shape = tx.textToShape(si1, si1Size, si1Size)
    gpusi1 = es.GPUShape().initBuffers()
    textPipeline.setupVAO(gpusi1)
    gpusi1.fillBuffers(si1Shape.vertices, si1Shape.indices, GL_STATIC_DRAW)
    gpusi1.texture = gpuText3DTexture
    si1Transform = tr.translate(0.05, 0.9, 0)
    si1Transformd = tr.translate(0.065, 0.915, 0)

    canhit = "Va a golpear?:"
    hitSize = 0.065
    hitShape = tx.textToShape(canhit, hitSize, hitSize)
    gpuHit = es.GPUShape().initBuffers()
    textPipeline.setupVAO(gpuHit)
    gpuHit.fillBuffers(hitShape.vertices, hitShape.indices, GL_STATIC_DRAW)
    gpuHit.texture = gpuText3DTexture
    hitTransform = tr.translate(-0.95, 0.8, 0)
    hitTransformd = tr.translate(-0.925, 0.815, 0)

    si2 = "si"
    si2Size = 0.065
    si2Shape = tx.textToShape(si2, si2Size, si2Size)
    gpusi2 = es.GPUShape().initBuffers()
    textPipeline.setupVAO(gpusi2)
    gpusi2.fillBuffers(si2Shape.vertices, si2Shape.indices, GL_STATIC_DRAW)
    gpusi2.texture = gpuText3DTexture
    si2Transform = tr.translate(0.0, 0.8, 0)
    si2Transformd = tr.translate(0.015, 0.815, 0)

    ######################################################

    # Colores para el mapa de calor

    color1 = [1.0, 0.0, 0.0]
    color2 = [0.5, 0.5, 0.0]
    color3 = [0.0, 1.0, 0.0]
    color4 = [0.0, 0.5, 0.5]
    color5 = [0.0, 0.0, 1.0]

    heatPipeline = es.SimpleTextureTransformShaderProgram()

    tabla = bs.createTextureQuad(1, 1)
    tablaShape = es.GPUShape().initBuffers()
    heatPipeline.setupVAO(tablaShape)
    tablaShape.fillBuffers(tabla.vertices, tabla.indices, GL_STATIC_DRAW)
    tablaShape.texture = es.textureSimpleSetup(
        getSpritePath("heat.png"), GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR)


    ######################################################

    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)
    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)
    t0 = glfw.get_time()

    # Application loop
    while not glfw.window_should_close(window):
        # Variables del tiempo
        t1 = glfw.get_time()
        delta = t1 -t0
        t0 = t1

        ######################################################

        # Chequeo de colisiones y cambio en el texto del HUD

        si1 = "si"
        si2 = "no"

        punta = np.array([controller.at[0] + 0.35*cos(controller.theta), controller.at[1] + 0.35*sin(controller.theta)])

        for i in range(len(ball)):
                if np.linalg.norm(ball[i].pos-punta) < 0.2:
                    si2 = "si"
                    if not controller.golpe and controller.action:
                        hit(ball[i])
                for j in range(i+1, len(ball)):
                    if areColliding(ball[i], ball[j]):
                        collide(ball[i], ball[j])
        controller.golpe = True

        # Contador de velocidades
        contador = 0

        # Movimiento de las pelotas y cambio en el texto del HUD
        for a in ball:          
            
            sg.findNode(balls,str(a.id)+"c").transform = tr.translate(a.pos[0], a.pos[1], -1.9625)
            sg.findNode(sombras,str(a.id)+"1").transform = tr.translate(a.pos[0], a.pos[1], -2.05)
            sg.findNode(sombras,str(a.id)+"2").transform = tr.matmul([tr.translate(0.0005*a.pos[0], 0.0005*a.pos[1], 0),tr.scale(1+abs(0.01*a.pos[0]),1+abs(0.01*a.pos[1]),1)])
           
            # Para evitar errores, las velocidades muy cercanas a 0 se igualan a 0:
            if abs(a.velocity[0]) < 0.002:
                a.velocity[0] = 0
            if abs(a.velocity[1]) < 0.002:
                a.velocity[1] = 0
           
            a.action(delta)
            collideWithBorder(a)

            contador += abs(a.velocity[0]) + abs(a.velocity[1])

            if abs(a.pos[0]) > 4.3 or abs(a.pos[1]) > 2.575:
                balls.childs.remove(sg.findNode(balls,str(a.id)+"c"))
                sombras.childs.remove(sg.findNode(sombras,str(a.id)+"2"))
                ball.remove(a)

        # Si la suma de las velocidades de las pelotas supera cierta cantidad, no se puede golpear 
        if contador < 0.05:
            controller.action = True
            contador = 0
        else:
            si1 = "no"            
            controller.action = False

        ######################################################

        # Golpe con el palo

        if controller.action:
            tiro.transform = tr.translate(0,0,0)
        if not controller.action:
            tiro.transform = tr.translate(-1.5,0,0)

        # Movimiento del palo

        palo.transform = tr.matmul([tr.translate(controller.eye[0],controller.eye[1],controller.eye[2]),tr.rotationX(pi/2),tr.rotationY(controller.theta),tr.uniformScale(0.65)])
        
        ######################################################

        # Actualización de la cámara sobre el palo

        atx = controller.eye[0] + np.cos(controller.theta)
        aty = controller.eye[1] + np.sin(controller.theta)
        atz = controller.eye[2]
        controller.at = np.array([atx, aty, atz])

        # Que cámara se va a utilizar
        if controller.camara:
            Matrix = tr.lookAt(feye,fat,fup)
        else:
            Matrix = tr.lookAt(controller.eye,controller.at,controller.up)

        ######################################################

        # Using GLFW to check for input events
        glfw.poll_events()

        # Setting up the projection transform
        projection = tr.perspective(60, float(width) / float(height), 0.1, 200)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        ######################################################

        # Vemos que shader usar en base a la configuracion

        if L == "On":
            Pipeline = sh.HeatPhongShaderProgram()
            heat = True
        else:
            Pipeline = TexPipeline
            heat = False

        ######################################################

        # Se definen parámetros distintos para cada grupo de objetos, y se dibujan respectivamente

        ######################################################

        # Sin iluminación

        # Dibujamos la habitación

        glUseProgram(roomPipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(roomPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(roomPipeline.shaderProgram, "view"), 1, GL_TRUE, Matrix)
        glUniformMatrix4fv(glGetUniformLocation(roomPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())

        if not heat:
            sg.drawSceneGraphNode(skybox, roomPipeline, "model")

        # Dibujamos las sombras
 
        glUseProgram(shadowPipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(shadowPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(shadowPipeline.shaderProgram, "view"), 1, GL_TRUE, Matrix)
        glUniformMatrix4fv(glGetUniformLocation(shadowPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())

        if not heat:
            sg.drawSceneGraphNode(sombras, shadowPipeline, "model")

        ######################################################
       
        # Con iluminación

        glUseProgram(Pipeline.shaderProgram)
 
        # Dibujamos la mesa y el palo

        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "La"), 0.25, 0.25, 0.25)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "Ld"), 0.35, 0.35, 0.35)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "Ls"), 0.7, 0.7, 0.7)

        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "Ka"), 0.8, 0.8, 0.8)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "Kd"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "Ks"), 0.7, 0.7, 0.7)

        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "lightPos"), 0, 0, 8)
        if heat:
            glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "lightPos"), 0, 0, 2.5)    
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "viewPosition"), 0, 0, 0)
        glUniform1ui(glGetUniformLocation(Pipeline.shaderProgram, "shininess"), 1)
        
        glUniform1f(glGetUniformLocation(Pipeline.shaderProgram, "constantAttenuation"), 0.001)
        glUniform1f(glGetUniformLocation(Pipeline.shaderProgram, "linearAttenuation"), 0.015)
        glUniform1f(glGetUniformLocation(Pipeline.shaderProgram, "quadraticAttenuation"), 0.01)

        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "color1"), color1[0], color1[1], color1[2])
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "color2"), color2[0], color2[1], color2[2])
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "color3"), color3[0], color3[1], color3[2])
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "color4"), color4[0], color4[1], color4[2])
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "color5"), color5[0], color5[1], color5[2])

        glUniformMatrix4fv(glGetUniformLocation(Pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(Pipeline.shaderProgram, "view"), 1, GL_TRUE, Matrix)
        glUniformMatrix4fv(glGetUniformLocation(Pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())

        sg.drawSceneGraphNode(tableNode, Pipeline, "model")
        sg.drawSceneGraphNode(stickt, Pipeline, "model")
        
        # Dibujamos las pelotas

        glUniform1ui(glGetUniformLocation(Pipeline.shaderProgram, "shininess"), 75)

        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "Ka"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "Kd"), 0.91, 0.91, 0.91)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "Ks"), 0.9, 0.9, 0.9)
        glUniformMatrix4fv(glGetUniformLocation(Pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())

        sg.drawSceneGraphNode(balls, Pipeline, "model")

        # Si el mapa de calor está encendido, dibujamos la tabla

        if heat:
            glUseProgram(heatPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(heatPipeline.shaderProgram, "transform"), 1, GL_TRUE, tr.matmul([tr.translate(0.0, -0.9, 0.0),tr.scale(1.0, 0.09, 1.0)]))
            heatPipeline.drawCall(tablaShape)

        # Dibujamos, si está encendido, el HUD

        if HUD == "On":
            glDisable(GL_DEPTH_TEST)

            glUseProgram(textPipeline.shaderProgram)
            glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "fontColor"), 0,0,0,1)
            glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "backColor"), 0,0,0,0)
            glUniformMatrix4fv(glGetUniformLocation(textPipeline.shaderProgram, "transform"), 1, GL_TRUE, pressTransform)
            textPipeline.drawCall(gpuPress)

            glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "fontColor"), 1,1,1,1)
            glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "backColor"), 0,0,0,0)
            glUniformMatrix4fv(glGetUniformLocation(textPipeline.shaderProgram, "transform"), 1, GL_TRUE, pressTransformd)
            textPipeline.drawCall(gpuPress)

            si1Shape = tx.textToShape(si1, si1Size, si1Size)
            gpusi1.fillBuffers(si1Shape.vertices, si1Shape.indices, GL_STATIC_DRAW)

            glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "fontColor"), 0,0,0,1)
            glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "backColor"), 0,0,0,0)
            glUniformMatrix4fv(glGetUniformLocation(textPipeline.shaderProgram, "transform"), 1, GL_TRUE, si1Transform)
            textPipeline.drawCall(gpusi1)

            if si1 == "si":
                glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "fontColor"), cos(10*t1),cos(pi*2/3+10*t1),cos(pi*4/3+10*t1),1)
            else:
                glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "fontColor"), 1,0,0,1)
            glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "backColor"), 0,0,0,0)
            glUniformMatrix4fv(glGetUniformLocation(textPipeline.shaderProgram, "transform"), 1, GL_TRUE, si1Transformd)
            textPipeline.drawCall(gpusi1)

            glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "fontColor"), 0,0,0,1)
            glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "backColor"), 0,0,0,0)
            glUniformMatrix4fv(glGetUniformLocation(textPipeline.shaderProgram, "transform"), 1, GL_TRUE, hitTransform)
            textPipeline.drawCall(gpuHit)

            glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "fontColor"), 1,1,1,1)
            glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "backColor"), 0,0,0,0)
            glUniformMatrix4fv(glGetUniformLocation(textPipeline.shaderProgram, "transform"), 1, GL_TRUE, hitTransformd)
            textPipeline.drawCall(gpuHit)

            si2Shape = tx.textToShape(si2, si2Size, si2Size)
            gpusi2.fillBuffers(si2Shape.vertices, si2Shape.indices, GL_STATIC_DRAW)

            glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "fontColor"), 0,0,0,1)
            glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "backColor"), 0,0,0,0)
            glUniformMatrix4fv(glGetUniformLocation(textPipeline.shaderProgram, "transform"), 1, GL_TRUE, si2Transform)
            textPipeline.drawCall(gpusi2)

            if si2 == "si":
                glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "fontColor"), cos(10*t1),cos(pi*2/3+10*t1),cos(pi*4/3+10*t1),1)
            else:
                glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "fontColor"), 1,0,0,1)
            glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "backColor"), 0,0,0,0)
            glUniformMatrix4fv(glGetUniformLocation(textPipeline.shaderProgram, "transform"), 1, GL_TRUE, si2Transformd)
            textPipeline.drawCall(gpusi2)

            glEnable(GL_DEPTH_TEST)

        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    glfw.terminate()