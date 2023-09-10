import random
import os
import re
from signal import raise_signal

# import basic pygame modules
import pygame as pg
import time
import json
from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Array
from http.server import HTTPServer, BaseHTTPRequestHandler

# see if we can load more than standard BMP
if not pg.image.get_extended():
    raise SystemExit("Sorry, extended image module required")

SCREENRECT = pg.Rect(0, 0, 1920, 1080)

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        pass

    def do_POST(self):
        if self.path == "/command":
            content_length = int(self.headers['Content-Length'])
            c.value = self.rfile.read(content_length)
        data = {}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

def run_http_server(port):     
    server = HTTPServer(('', port), RequestHandler)
    c.value = b''
    print('http server running')
    server.serve_forever()

def load_image(file):
    try:
        surface = pg.image.load(file)
    except pg.error:
        raise SystemExit('Could not load image "%s" %s' % (file, pg.get_error()))
    return surface.convert()

class Player(pg.sprite.Sprite):
    images = None

    def __init__(self):
        pg.sprite.Sprite.__init__(self, self.containers)
        self.image = load_image('spot.png')
        self.rect = self.image.get_rect(midbottom=SCREENRECT.midbottom)

    def move(self, x, y):
        self.rect = self.image.get_rect().move(40+50+x*250, 40+50+y*250)

class Status1(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.font = pg.font.Font('AaShiSongTi-2.ttf', 235)
        self.update()
        self.rect = self.image.get_rect().move(1080, 240)

    def reflash(self, h):
        if h == 's':
            key1 = '单手'
        elif h == 'd':
            key1 = '双手'
        elif h == 'm':
            key1 = '单/双手'
        else:
            raise NotImplementedError
         
        self.image = self.font.render('{}'.format(key1), 1, pg.Color("black"))
  
    def update(self):
        self.image = self.font.render('', 1, pg.Color("black"))

class Status2(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.font = pg.font.Font('AaShiSongTi-2.ttf', 235)
        self.update()
        self.rect = self.image.get_rect().move(1080, 500)

    def reflash(self, a):
        if a == 'p':
            key2 = '抛'
        elif a == 'j':
            key2 = '接'
        else:
            raise NotImplementedError        
        
        self.image = self.font.render('{}'.format(key2), 1, pg.Color("black"))
  
    def update(self):
        self.image = self.font.render('', 1, pg.Color("black"))


def main(winstyle=0):
    # Initialize pygame
    pg.init()

    fullscreen = False
    # Set the display mode
    winstyle = 0  # |FULLSCREEN
    bestdepth = pg.display.mode_ok(SCREENRECT.size, winstyle, 32)
    screen = pg.display.set_mode(SCREENRECT.size, winstyle, bestdepth)

    # decorate the game window
    pg.display.set_caption("Remote Display")
    pg.mouse.set_visible(0)

    # # create the background, tile the bgd image
    bgdtile = load_image('background.png')
    background = pg.Surface(SCREENRECT.size)
    for x in range(0, SCREENRECT.width, bgdtile.get_width()):
        background.blit(bgdtile, (x, 0))
    screen.blit(background, (0, 0))
    pg.display.flip()

    all = pg.sprite.RenderUpdates()

    # assign default groups to each sprite class
    Player.containers = all  
    Status1.containers = all
    Status2.containers = all
    
    clock = pg.time.Clock()

    # # initialize our starting sprites
    player = Player()
    if pg.font:
        s1 = Status1()
        s2 = Status2()
        all.add(s1)
        all.add(s2)

    bb = False

    # Run our main loop whilst the player is alive.
    while player.alive():
        if len(c.value) > 0 and bb:
            bgdtile = load_image('background.png')
            background = pg.Surface(SCREENRECT.size)
            for x in range(0, SCREENRECT.width, bgdtile.get_width()):
                background.blit(bgdtile, (x, 0))
            screen.blit(background, (0, 0))
            pg.display.flip()
            bb = False
        
        # clear/erase the last drawn sprites
        all.clear(screen, background)
        all.update()

        if len(c.value) > 0:
            l = re.split('[,]', c.value.decode("utf-8").replace(" ", ""))
            x, y, h, a = int(l[0]), int(l[1]), l[2], l[3]
            player.move(x, y)
            s1.reflash(h)
            s2.reflash(a)

        if len(c.value) > 0:
            dirty = all.draw(screen)
            pg.display.update(dirty)
        else:
            screen.fill((0, 0, 0))
            pg.display.update()
            bb = True

        # cap the framerate at 40fps. Also called 40HZ or 40 times per second.
        clock.tick(40)

    pg.time.wait(1000)


# call the "main" function if running this script
if __name__ == "__main__":
    lock = Lock()
    c = Array('c', b' '*100, lock=lock)
    p = Process(target=run_http_server, args=(5000,))
    p.start()
    time.sleep(1)
    main(c)
    pg.quit()
    p.join()