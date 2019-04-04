from tkinter import Tk, Button
from tkinter.filedialog import askopenfilename
import cv2
from PIL import Image, ImageTk

Tk().withdraw()
filename = askopenfilename(filetypes=(("bmp file", "*.bmp"),("jpg file", "*.jpg"),("All Files", "*.*")))
# print(filename)
# img = Image.open(filename)
# img.show()
# # img_rgb = cv2.imread(filename)
# root = Tk()
# image = ImageTk.PhotoImage(file="norm10.bmp")
# Button(root, image=image, command=lambda: print('click')).pack()
# root.mainloop()

# import tkinter
# import Image, ImageTk, ImageDraw
#
# image_file = "norm1.jpg"
#
# w = tkinter.Tk()
#
# img = Image.open(image_file)
# width, height = img.size
# ca = tkinter.Canvas(w, width=width, height=height)
# ca.pack()
# photoimg = ImageTk.PhotoImage("RGB", img.size)
# photoimg.paste(img)
# ca.create_image(width//2,height//2, image=photoimg)
# tkinter.mainloop()

import pygame, sys
from PIL import Image

pygame.init()


def displayImage(screen, px, topleft, prior):
    # ensure that the rect always has positive width, height
    x, y = topleft
    width = pygame.mouse.get_pos()[0] - topleft[0]
    height = pygame.mouse.get_pos()[1] - topleft[1]
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current

    # draw transparent box and blit it onto canvas
    screen.blit(px, px.get_rect())
    im = pygame.Surface((width, height))
    im.fill((128, 128, 128))
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    im.set_alpha(128)
    screen.blit(im, (x, y))
    pygame.display.flip()

    # return current box extents
    return (x, y, width, height)


def setup(path):
    px = pygame.image.load(path)
    screen = pygame.display.set_mode(px.get_rect()[2:])
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px


def mainLoop(screen, px):
    topleft = bottomright = prior = None
    n = 0
    while n != 1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    n = 1
        if topleft:
            prior = displayImage(screen, px, topleft, prior)
    return (topleft + bottomright)


if __name__ == "__main__":
    input_loc = filename
    output_loc = 'out.bmp'
    screen, px = setup(input_loc)
    left, upper, right, lower = mainLoop(screen, px)

    # ensure output rect always has positive width, height
    if right < left:
        left, right = right, left
    if lower < upper:
        lower, upper = upper, lower
    im = Image.open(input_loc)
    im = im.crop((left, upper, right, lower))
    pygame.display.quit()
    im.save(output_loc)
