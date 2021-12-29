
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

import cv2
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera

from kivy.uix.label import Label
#from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
##new
from kivy.uix.filechooser import  FileChooserIconView

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
        self.page = GridLayout(padding=10)
        self.page.cols = 1
        
        self.img=Image(source='imgs/insight.jpg')
        
        self.page.add_widget(self.img)

        self.button_cam=Button(text="Camera", font_size=40)
        self.button_cam.bind(on_press=lambda x: self.to_screen('Camera'))
        self.page.add_widget(self.button_cam)

        self.button_pt=Button(text="Upload photo", font_size=40)
        self.button_pt.bind(on_press=lambda x: self.to_screen('File'))
        self.page.add_widget(self.button_pt)
        
        #self.page.add_widget(Label(text='Chatbot'))

        self.chatbox=TextInput(text='사용할 이름을 입력하세요.',multiline=False)
        self.chatbox.font_name='font/malgun.ttf'
        self.page.add_widget(self.chatbox) 

        self.chatbox.bind(on_text_validate=self.on_enter, focus=self.on_focus)
        self.add_widget(self.page)

    def on_enter(self, *args):
        #self.chatbox.text += "\n입력이 완료되었습니다.\n "
        #print(self.chatbox.text)
        name=get_var()
        name=self.chatbox.text
        print(name,"in HomeScreen")
        return name

    def on_focus(self, instance, value):
        self.chatbox.focus = True
          
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


### 파일 탐색기로 선택한 이미지 경로 저장해서 캡셔닝에서 사용
class FileScreen(CaptionScreen):
    def __init__(self, **kwargs):
        super(FileScreen, self).__init__(**kwargs)
        self.page = BoxLayout(orientation='vertical',padding=10, spacing=20)

        ### 클릭한 이미지가 여기에 뜸
        self.my_image=Image(source="")
        self.page.add_widget(self.my_image)

        ### 파일 탐색기
        self.fichoo = FileChooserIconView(size_hint_y = 0.8)
        self.fichoo.dirselect=True

        ### 더블클릭 할 때마다 selected 호출됨
        self.fichoo.bind(selection=self.selected)
        self.page.add_widget(self.fichoo)

        ### 홈 버튼
        self.button_home=Button(text="Home", size_hint=(.5, .5))
        self.button_home.bind(on_press=lambda x: self.to_screen('Home'))
        self.page.add_widget(self.button_home)

        ### 결과 버튼 (밑에서 최종 파일 클릭 후, select 버튼 누른 후에 눌러야함)
        self.button_res=Button(text="OK", size_hint=(.5, .5))
        self.button_res.bind(on_press=lambda x: self.to_screen('Result'))
        self.page.add_widget(self.button_res)

        self.add_widget(self.page)
    
    
    def selected(self, selection,val):
        ### 위에 비워놓은 이미지에 클릭한 파일 이미지 띄우는 시도.
        ### why '시도'? -> /Users/iyunju/Documents 이렇게 최종 파일이 선택되지 않은 상태면
        ### 에러 메세지 출력되고, 최종 선택한 파일 경로를 알 수 없음
        ### (경로 경로 끝에 최종으로 이미지 파일을 더블클릭했을 때 제대로 띄워지고, 그게 최종 경로)
        if self.fichoo.selection[0][-3:] in ['png','jpg']:
            self.my_image.source=self.fichoo.selection[0]
        else:
            self.my_image.source="imgs/mansoo.jpeg"
        ### 그래서 사용자가 수동으로,, 사용할 이미지 더블클릭 후 이 버튼 누르면 final로 넘어감
        self.bt=Button(text="Choose", size_hint=(.1, .1))
        self.bt.bind(on_press=self.final)
        self.add_widget(self.bt)

    ### 선택한 파일이 최종 선택 파일이기 때문에 이 경로 저장 (지금은 print로 해놓음)
    ### 사용자는 select 눌러서 여기에 경로 넘겨준 후 Go to Result 버튼 눌러야 함.
    def final(self,val):
        image = cv2.imread(self.fichoo.selection[0], cv2.IMREAD_COLOR)
        self.image_path='imgs/pic.png'
        #print("now in file",i)
        self.save_path='imgs/pic_captioned.png'
        #self.save_path=save_path+str(i)+'.png'
        
        cv2.imwrite(self.image_path,image)
        name=get_var()
        print(name,"in FileScreen")
        test(self.image_path,self.save_path, True, name)
        print(self.fichoo.selection[0])

    

    #def refresh(self):
    #    self.clear_widgets([self.manager.get_screen('Result')])
        #screen={}
        #screen['Result'] = ResultScreen(name='Result')
     #   self.add_widget(ResultScreen(name='Result'))

class CameraScreen(CaptionScreen):
    
    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
    
        self.page = GridLayout(padding=10)
        self.page.cols = 1

        self.img1=Image()
        self.page.add_widget(self.img1)

        self.save_button=Button(text='Capture!',size_hint=(.5, .5))
        self.save_button.bind(on_press=self.take_picture)
        self.page.add_widget(self.save_button)

        self.button_home=Button(text="Home", size_hint=(.5, .5))
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
        
        test(self.image_path,self.save_path, True)
        
        self.to_screen('Result')
        #####       로딩팝업을 띄운다          #####
        #content=BoxLayout(orientation='vertical')
        #content.add_widget(Label(text='Loading...'))
        #popup=Popup(title='Loading...', content=content, 
                       # auto_dismiss=True, size_hint=(.5,.5))
        #popup.open()
        
        # 출력된 파일들 경로에 저장
        
        #####    끝나면 pop up 닫기          #####
       # popup.dismiss()
        #####      다음 화면으로 이동          #####


        
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

        self.page = GridLayout(padding=10)
        self.page.cols = 1

        self.image=Image(source='imgs/pic_captioned.png')
        self.image.reload()
        
        self.button_sv=Button(text="Save", font_size=40)
        self.button_sv.bind(on_press=self.on_pressed_sv)

        self.button_rt1=Button(text="Retry Camera", font_size=40)
        self.button_rt1.bind(on_press=lambda x: self.to_screen('Camera'))
        
        self.button_rt2=Button(text="Retry Album", font_size=40)
        self.button_rt2.bind(on_press=lambda x: self.to_screen('File'))
        
        self.button_pt=Button(text="Home", font_size=40)
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

    def on_pressed_sv(self):
        ###### 임시 파일 이미지를 저장 ######
        pass