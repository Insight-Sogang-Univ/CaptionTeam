
from logging import ERROR
import kivy
kivy.require('2.0.0')
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
import os

#### for 카메라
import cv2
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from jnius import autoclass

#### for 파일 탐색
from kivy.uix.label import Label
#from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

from kivy.uix.filechooser import  FileChooserIconView
from kivy.utils import platform

#from camera import KivyCamera
from config import *

from kivy.core.window import Window
from kivy.lang import Builder

from test import test

### name이 전체에서 불러올 수 있게 수정해야함
global name
name=None
class CaptionScreen(Screen):
    def __init__(self, **kwargs):
        super(CaptionScreen, self).__init__(**kwargs)
       
    def to_screen(self, scr_name):
        self.manager.get_screen(scr_name).update_layout()
        self.manager.current = scr_name
    
    def update_layout(self):
        pass

def get_var():
    global name
    return name

class HomeScreen(CaptionScreen):
    def __init__(self, **kwargs):
        super(HomeScreen, self).__init__(**kwargs)
        Window.size = (400, 700)
        self.page = GridLayout(padding=10)
        self.page.cols = 1
        
        self.lb=Label(text='캡숑!캡셔닝 APP', font_size=40,size_hint=(1, .1),
                        font_name='font/RixYeoljeongdo Regular.ttf')
        self.page.add_widget(self.lb)

        self.img=Image(source='imgs/insight.jpg', size_hint =(1, .7))
        
        self.page.add_widget(self.img)
        
        self.chatbox=TextInput(text='사용할 이름을 입력하세요 \n =>',multiline=False, font_size=60,
                                size_hint =(.5, .3),pos =(20, 20))
        self.chatbox.font_name='font/RixYeoljeongdo Regular.ttf'
        self.page.add_widget(self.chatbox) 
        
        self.button_cam=Button(text="사진 찍기", font_size=40, 
                                size_hint =(.5, .2),pos =(20, 20),
                                font_name='font/RixYeoljeongdo Regular.ttf')
        self.button_cam.bind(on_press=lambda x: self.to_screen('Camera'))
        self.page.add_widget(self.button_cam)

        self.button_pt=Button(text="앨범 불러오기", font_size=40,
                                size_hint =(.5, .2),pos =(20, 20),
                                font_name='font/RixYeoljeongdo Regular.ttf')
        self.button_pt.bind(on_press=lambda x: self.to_screen('File'))
        self.page.add_widget(self.button_pt)

        self.chatbox.bind(on_text_validate=self.on_enter, focus=self.on_focus)
        self.add_widget(self.page)

    def on_enter(self, *args):
        name=get_var()
        name=self.chatbox.text
        print(name,"in HomeScreen")
        return name

    def on_focus(self, instance, value):
        self.chatbox.focus = True
          
    ##로딩 화면
    def next(self, *args): ##lambda함수에서 전달 받은 args
    
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


### 파일 탐색기로 선택한 이미지 경로 저장해서 캡셔닝에서 사용
class FileScreen(CaptionScreen):
    def __init__(self, **kwargs):
        super(FileScreen, self).__init__(**kwargs)

        self.page = BoxLayout(orientation='vertical',padding=10, spacing=20)
        self.lb=Label(text="미리보기", font_size=40, size_hint=(1,.1), font_name='font/RixYeoljeongdo Regular.ttf')
        self.page.add_widget(self.lb)
        ### 클릭한 이미지가 여기에 뜸
        self.my_image=Image(source="imgs/insight.jpg",size_hint =(1, .5))
        self.page.add_widget(self.my_image)

        ### 파일 탐색기
        self.fichoo = FileChooserIconView(size_hint_y = 0.6)
        self.fichoo.dirselect=True

        ### 더블클릭 할 때마다 selected 호출됨
       
        self.fichoo.bind(selection=self.selected)
        self.page.add_widget(self.fichoo)
    
        ### 결과 버튼 (밑에서 최종 파일 클릭 후, select 버튼 누른 후에 눌러야함)
        self.button_res=Button(text="결과 보기", font_size=40, size_hint =(1, .15),pos =(20, 20),
                            font_name='font/RixYeoljeongdo Regular.ttf')
        self.button_res.bind(on_press=self.final)
        self.page.add_widget(self.button_res)

         ### 홈 버튼
        self.button_home=Button(text="홈으로", font_size=40, size_hint =(1, .15),pos =(20, 20),
                                font_name='font/RixYeoljeongdo Regular.ttf')
        self.button_home.bind(on_press=lambda x: self.to_screen('Home'))
        self.page.add_widget(self.button_home)

        self.add_widget(self.page)

    
    def selected(self, selection,val):
        if self.fichoo.selection[0][-3:] in ['png','jpg']:
            self.my_image.source=self.fichoo.selection[0]
            #self.my_image.reload()
            image = cv2.imread(self.fichoo.selection[0], cv2.IMREAD_COLOR)
            self.image_path='imgs/pic.png'
            self.save_path='imgs/pic_captioned.png'
            
            cv2.imwrite(self.image_path,image)
            print(self.fichoo.selection[0])

        else:
            self.my_image.source="imgs/mansoo.jpeg"


    def final(self,val):
        test(self.image_path,self.save_path, True)
        self.to_screen('Result')


class CameraScreen(CaptionScreen):
    
    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
    
        self.page = GridLayout(padding=10)
        self.page.cols = 1

        self.lb=Label(text="사진을 찍어보세요!", font_size=40, font_name='font/RixYeoljeongdo Regular.ttf',
                            size_hint=(1,.1), pos=(20,20))
        self.page.add_widget(self.lb)

        self.img1=Image(size_hint=(1,.6))
        self.page.add_widget(self.img1)

        self.save_button=Button(text='눌러서 촬영', font_size=40,size_hint =(1, .2),pos =(20, 20),
                                font_name='font/RixYeoljeongdo Regular.ttf')
        self.save_button.bind(on_press=self.take_picture)
        self.page.add_widget(self.save_button)

        self.button_home=Button(text="홈으로", font_size=40, size_hint =(1, .2),pos =(20, 20),
                        font_name='font/RixYeoljeongdo Regular.ttf')
        self.button_home.bind(on_press=lambda x: self.to_screen('Home'))
        self.page.add_widget(self.button_home)
        
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        self.add_widget(self.page)

    def update(self, *args):
        ret, frame = self.capture.read()
        self.image_frame=frame
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.img1.texture = texture

    def take_picture(self, instance):
        ##### 사진을 찍고 임시 파일로 저장한다 #####
        self.image_path='imgs/pic.png'
        self.save_path='imgs/pic_captioned.png'
        #print("now in camera",i)
        #self.save_path=save_path+str(i)+'.png'
        name=get_var()
        print(name,"in CameraScreen")

        cv2.imwrite(self.image_path, self.image_frame,name)
        
        #####    임시 파일을 모델에 입력한다   #####
        
        test(img_path=self.image_path,save_path=self.save_path, save= True)
        
        self.to_screen('Result')


        
class ResultScreen(CaptionScreen):
    def __init__(self, **kwargs):
        super(ResultScreen, self).__init__(**kwargs)
        ### 얘네가 그때그때 실행되기ㅔ 어딘가에 추가 시켜서 그때그때 다시 실행되게.
        ### 함수 추가해보자

        self.path='imgs/pic_captioned.png'
        #image_path=path+str(i)+'.png'
        
        self.update_layout()

    def update_layout(self):
        self.clear_widgets()

        self.path='imgs/pic_captioned.png'
        self.save_path='imgs/captioned_result.png'

        self.page = GridLayout(padding=10)
        self.page.cols = 1

        self.image=Image(source=self.path, size_hint=(1,.6), pos=(20,20),allow_stretch= True)
        self.image.reload()
        
        self.button_sv=Button(text="저장", font_size=40, size_hint=(1,.1),
                        font_name='font/RixYeoljeongdo Regular.ttf')
        self.button_sv.bind(on_press=self.on_pressed_sv)

        self.button_rt1=Button(text="카메라로 이동", font_size=40, size_hint=(1,.1),
                        font_name='font/RixYeoljeongdo Regular.ttf')
        self.button_rt1.bind(on_press=lambda x: self.to_screen('Camera'))
        
        self.button_rt2=Button(text="앨범으로 이동", font_size=40, size_hint=(1,.1),
                        font_name='font/RixYeoljeongdo Regular.ttf')
        self.button_rt2.bind(on_press=lambda x: self.to_screen('File'))
        
        self.button_pt=Button(text="홈으로", font_size=40, size_hint=(1,.1),
                        font_name='font/RixYeoljeongdo Regular.ttf')
        self.button_pt.bind(on_press=lambda x: self.to_screen('Home'))

        self.page.add_widget(self.image)
        self.page.add_widget(self.button_sv)
        self.page.add_widget(self.button_rt1)
        self.page.add_widget(self.button_rt2)
        self.page.add_widget(self.button_pt)

        self.add_widget(self.page)


    def get_image(self, *args):
        image_path='imgs/pic_captioned.png'
        img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        cv2.imshow('Result image',img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    def on_pressed_sv(self, *args):
        ###### 임시 파일 이미지를 저장 ######
        image = cv2.imread(self.path, cv2.IMREAD_COLOR)
        cv2.imwrite(self.save_path,image)