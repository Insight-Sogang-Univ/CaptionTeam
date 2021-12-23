import cv2

import kivy
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera

# pc에서는 작동하는 버전
class KivyCamera:
    def __init__(self):
        self.layout=BoxLayout(orientation='vertical')
        self.img1=Image()
        self.layout.add_widget(self.img1)

        self.save_img_button=Button(text='Click Here!',size_hint=(.5, .5))
        self.save_img_button.bind(on_press=self.take_picture)
        self.layout.add_widget(self.save_img_button)
        
        self.close_button=Button(text='Close',size_hint=(.5, .5))
        self.layout.add_widget(self.close_button)
        self.close_button.bind(on_press=self.close_app)
        
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

    def update(self, *args):
        ret, frame = self.capture.read()
        self.image_frame=frame
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.img1.texture = texture

    def take_picture(self, *args):
        imange_name='pic.png'
        cv2.imwrite(imange_name, self.image_frame)

    def close_app(self, *args):
        #args[0].dismiss()
        self.remove_widget(self.img1)
        self.remove_widget(self.save_img_button)
        self.remove_widget(self.close_button)