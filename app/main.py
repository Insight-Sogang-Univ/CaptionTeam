from typing import Text

import kivy
kivy.require('2.0.0') # replace with your current kivy version !

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import ScreenManager, NoTransition
from kivy.properties import ObjectProperty, NumericProperty


from screens import *

class CaptionApp(App):
    def build(self):
        # Create the manager
        sm = ScreenManager()
        
        # Create Screens
        screen = {}
        screen['Home'] = HomeScreen(name='Home')
        screen['Camera'] = CameraScreen(name='Camera')
        screen['Result'] = ResultScreen(name='Result')
        screen['File'] = FileScreen(name='File')

        # Add Screens
        for scr in screen.values():
            sm.add_widget(scr)
        
        # Start from Home
        sm.current = 'Home'
        
        return sm

if __name__=='__main__':
    CaptionApp().run()