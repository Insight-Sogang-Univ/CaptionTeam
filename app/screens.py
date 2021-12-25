
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
#from pract import Example
import cv2
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera

##new
from kivy.uix.filechooser import FileChooser, FileChooserIconView, FileChooserListView
from kivy.uix.textinput import TextInput

#from camera import KivyCamera
from config import *

from kivy.core.window import Window
from kivy.lang import Builder

#from kivymd.app import MDApp
#from kivymd.uix.filemanager import MDFileManager
#from kivymd.toast import toast


class CaptionScreen(Screen):
    def __init__(self, **kwargs):
        super(CaptionScreen, self).__init__(**kwargs)
        
    def to_screen(self, scr_name):
        self.manager.current = scr_name

class HomeScreen(CaptionScreen):
    def __init__(self, **kwargs):
        super(HomeScreen, self).__init__(**kwargs)
        self.page = GridLayout(padding=10)
        self.page.cols = 1
        
        self.img=Image(source='insight.jpg')
        
        self.page.add_widget(self.img)

        self.button_cam=Button(text="Camera", font_size=40)
        self.button_cam.bind(on_press=lambda x: self.to_screen('Camera'))
        self.page.add_widget(self.button_cam)

        self.button_pt=Button(text="Upload photo", font_size=40)
        self.button_pt.bind(on_press=lambda x: self.to_screen('File'))
        self.page.add_widget(self.button_pt)
        
        self.add_widget(self.page)
        
    
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
        self.button_res=Button(text="Go to Result", size_hint=(.5, .5))
        self.button_res.bind(on_press=lambda x: self.to_screen('Result'))
        self.page.add_widget(self.button_res)

        self.add_widget(self.page)
    
    
    def selected(self, selection,val):
        ### 위에 비워놓은 이미지에 클릭한 파일 이미지 띄우는 시도.
        ### why '시도'? -> /Users/iyunju/Documents 이렇게 최종 파일이 선택되지 않은 상태면
        ### 에러 메세지 출력되고, 최종 선택한 파일 경로를 알 수 없음
        ### (경로 경로 끝에 최종으로 이미지 파일을 더블클릭했을 때 제대로 띄워지고, 그게 최종 경로)
        self.my_image.source=self.fichoo.selection[0]
        ### 그래서 사용자가 수동으로,, 사용할 이미지 더블클릭 후 이 버튼 누르면 test로 넘어감
        self.bt=Button(text="Select", size_hint=(.1, .1))
        self.bt.bind(on_press=self.test)
        self.add_widget(self.bt)

    ### 선택한 파일이 최종 선택 파일이기 때문에 이 경로 저장 (지금은 print로 해놓음)
    ### 사용자는 select 눌러서 여기에 경로 넘겨준 후 Go to Result 버튼 눌러야 함.
    def test(self,val):
        print(self.fichoo.selection[0])


class CameraScreen(CaptionScreen):
    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
        #self.camera = KivyCamera()
        #KivyCamera()
        #popup=Popup(content=KivyCamera, auto_dismiss=False, size_hint=(.5,.5))
        #popup.open()
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
        ##### 사진을 찍고 임시 파일로 저장한다 ##### --> 경로 설정해야 할 필요 있을지도..
        imange_name='pic.png'
        cv2.imwrite(imange_name, self.image_frame)
        
        #####       로딩팝업을 띄운다          #####
        content=BoxLayout(orientation='vertical')
        content.add_widget(Label(text='Loading...'))
        popup=Popup(title='Loading...', content=content, 
                        auto_dismiss=True, size_hint=(.5,.5))
        popup.open()
        
        #####    임시 파일을 모델에 입력한다   #####
        
        # 출력된 파일들 경로에 저장
    
        
        #####    끝나면 pop up 닫기          #####
        popup.dismiss()
        #####      다음 화면으로 이동          #####
        ##### 12.24아직까진 이 과정이 빛의 속도로 지나감 #####
        self.to_screen('Result')
        
        
class ResultScreen(CaptionScreen):
    def __init__(self, **kwargs):
        super(ResultScreen, self).__init__(**kwargs)
        self.page = GridLayout(padding=10)
        self.page.cols = 1
        
        self.img = Image(source=IMG_SAVE_PATH)
        self.page.add_widget(self.img)

        self.button_sv=Button(text="Save", font_size=40)
        self.button_sv.bind(on_press=self.on_pressed_sv)
        self.page.add_widget(self.button_sv)

        self.button_rt1=Button(text="Retry Camera", font_size=40)
        self.button_rt1.bind(on_press=lambda x: self.to_screen('Camera'))
        self.page.add_widget(self.button_rt1)

        self.button_rt2=Button(text="Retry Album", font_size=40)
        self.button_rt2.bind(on_press=lambda x: self.to_screen('File'))
        self.page.add_widget(self.button_rt2)

        self.button_pt=Button(text="Home", font_size=40)
        self.button_pt.bind(on_press=lambda x: self.to_screen('Home'))
        self.page.add_widget(self.button_pt)
        
        self.add_widget(self.page)
        
    def on_pressed_sv(self):
        ###### 임시 파일 이미지를 저장 ######
        pass