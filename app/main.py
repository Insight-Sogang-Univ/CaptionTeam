import cv2
from typing import Text

import kivy
kivy.require('2.0.0') # replace with your current kivy version !

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition
from kivy.properties import ObjectProperty, NumericProperty

from camera_pc import KivyCamera

class MyFirstScreen(GridLayout):
    def __init__(self, camera, **kwargs):
        super(MyFirstScreen, self).__init__(**kwargs)
        self.camera = camera
        
        self.cols=1

        self.img=Image(source='mansoo.jpeg')
        self.add_widget(self.img)

        self.button_cam=Button(text="Camera", font_size=40)
        self.button_cam.bind(on_press=self.on_pressed_cam)
        self.add_widget(self.button_cam)

        self.button_pt=Button(text="Upload photo", font_size=40)
        self.button_pt.bind(on_press=self.on_pressed_pt)
        self.add_widget(self.button_pt)

    def on_pressed_cam(self, instance):
        #KivyCamera().run()
        
        popup=Popup(content=self.camera.layout)
        #popup=Popup(content=KivyCamera().run())
        popup.open()
        
        #KivyCamera().stop()

    def on_pressed_pt(self, instance):
        content=BoxLayout(orientation='vertical')
        pt_button=Button(text='Choose picture')
        content.add_widget(pt_button)
        ## auto_dismiss True이면 저절로 팝업 없어짐, False면 수동, 팝업 되는 대상이 content라는 위젯 자체
        popup=Popup(title='Photo Album', content=content, 
                        auto_dismiss=False, size_hint=(.5,.5))
            ## 팝업 오픈
        popup.open()
            ## 한번만 쓰고 버리는 람다함수를 on_press로 전달
            ## press button 됐을 때, restart_board 실행됨. pop up 닫기 위함.
        pt_button.bind(on_press=lambda *args:self.next(popup, *args))
    
    ##로딩 화면
    def next(self, *args): ##lambda함수에서 전달 받은 args
        #print("args",args)
        ## 0번째 버튼 args가 닫을 args.
        args[0].dismiss() ##팝업창 닫힘
        
        levels=BoxLayout(orientation='vertical')
        lv_button=Button(text='See the result')
        levels.add_widget(lv_button)
        ## auto_dismiss True이면 저절로 팝업 없어짐, False면 수동, 팝업 되는 대상이 content라는 위젯 자체
        popups=Popup(title='Loading...', content=levels,
                        auto_dismiss=False, size_hint=(.5,.5))
            ## 팝업 오픈
        popups.open()
            ## 한번만 쓰고 버리는 람다함수를 on_press로 전달
            ## press button 됐을 때, restart_board 실행됨. pop up 닫기 위함.
        lv_button.bind(on_press=lambda *args:self.restart(popups, *args))
    
    ##결과 화면
    def restart(self, *args):
        args[0].dismiss() #전 팝업창 닫힘

        self.respic=Image(source='mansoo.jpeg')
        self.add_widget(self.respic)
        self.add_widget(Label(text='Yunju is smiling at the beach.'))
        
        self.button_dw=Button(text="Download", font_size=40)
        self.button_dw.bind(on_press=self.on_pressed_dw)
        self.add_widget(self.button_dw)

        self.button_hm=Button(text="Home", font_size=40)
        self.button_hm.bind(on_press=self.__init__)
        self.add_widget(self.button_hm)

        self.remove_widget(self.img)
        self.remove_widget(self.button_pt)
        self.remove_widget(self.button_cam)

    def on_pressed_dw(self, instance):
        content=BoxLayout(orientation='vertical')
        #cam_button=Button(text='downloading...')
        #content.add_widget(cam_button)
        ## auto_dismiss True이면 저절로 팝업 없어짐, False면 수동, 팝업 되는 대상이 content라는 위젯 자체
        popup=Popup(title='Downloading...', content=content, 
                        auto_dismiss=True, size_hint=(.5,.5))
            ## 팝업 오픈
        popup.open()
            ## 한번만 쓰고 버리는 람다함수를 on_press로 전달
            ## press button 됐을 때, restart_board 실행됨. pop up 닫기 위함.
        #cam_button.bind(on_press=lambda *args:self.next(popup, *args))


    def on_pressed_hm(self, instance):
        '''
        초기화면으로 가자~
        '''
        content=BoxLayout(orientation='vertical')
        cam_button=Button(text='Click here!')
        content.add_widget(cam_button)
        ## auto_dismiss True이면 저절로 팝업 없어짐, False면 수동, 팝업 되는 대상이 content라는 위젯 자체
        popup=Popup(title='Camera App', content=content, 
                        auto_dismiss=False, size_hint=(.5,.5))
            ## 팝업 오픈
        popup.open()
            ## 한번만 쓰고 버리는 람다함수를 on_press로 전달
            ## press button 됐을 때, restart_board 실행됨. pop up 닫기 위함.
        cam_button.bind(on_press=lambda *args:self.next(popup, *args))

        self.remove_widget(self.img)
        self.remove_widget(self.button_pt)
        self.remove_widget(self.button_cam)

class MyApp(App):
    def build(self):
        camera = KivyCamera()
        return MyFirstScreen(camera) ## 혹은 여기

if __name__=='__main__':
    MyApp().run() ## 여기서 초기화면을?