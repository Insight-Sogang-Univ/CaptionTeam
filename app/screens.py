
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen

from camera import KivyCamera
from config import *

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
        
        self.img=Image(source='mansoo.jpeg')
        
        self.page.add_widget(self.img)

        self.button_cam=Button(text="Camera", font_size=40)
        self.button_cam.bind(on_press=lambda x: self.to_screen('Camera'))
        self.page.add_widget(self.button_cam)

        self.button_pt=Button(text="Upload photo", font_size=40)
        self.button_pt.bind(on_press=self.on_pressed_pt)
        self.page.add_widget(self.button_pt)
        
        self.add_widget(self.page)
        
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

class CameraScreen(CaptionScreen):
    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
        self.camera = KivyCamera()
        
        self.page = GridLayout(padding=10)
        self.page.cols = 1
        
        self.save_button=Button(text='Capture!',size_hint=(.5, .5))
        self.save_button.bind(on_press=self.take_picture)
        self.page.add_widget(self.save_button)
        
        self.home_button=Button(text='Back',size_hint=(.5, .5))
        self.page.add_widget(self.home_button)
        
        self.add_widget(self.page)
        
    def take_picture(self, instance):
        ##### 사진을 찍고 임시 파일로 저장한다 #####
        
        
        #####       로딩팝업을 띄운다          #####
        content=BoxLayout(orientation='vertical')
        content.add_widget(Label(text='Loading...'))
        popup=Popup(title='Loading...', content=content, 
                        auto_dismiss=True, size_hint=(.5,.5))
        popup.open()
        
        #####    임시 파일을 모델에 입력한다   #####
        
        # 출력된 파일들 경로에 저장
    
        
        #####    끝나면 pop up 닫기           #####
        popup.dismiss()
        #####      다음 화면으로 이동          #####
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

        self.button_rt=Button(text="Retry", font_size=40)
        self.button_rt.bind(on_press=lambda x: self.to_screen('Camera'))
        self.page.add_widget(self.button_rt)

        self.button_pt=Button(text="Home", font_size=40)
        self.button_pt.bind(on_press=lambda x: self.to_screen('Home'))
        self.page.add_widget(self.button_pt)
        
        self.add_widget(self.page)
        
    def on_pressed_sv(self):
        ###### 임시 파일 이미지를 저장 ######
        pass