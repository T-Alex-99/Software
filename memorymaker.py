from ctypes import alignment
from random import randint
import tkinter
import tkinter.messagebox
from turtle import bgcolor
import customtkinter

import cv2  # still used to save images out
import numpy as np
from decord import VideoReader
from decord import cpu, gpu

import os

from tkinter import ANCHOR, W, filedialog
from PIL import ImageTk, Image


from model import predict_images
#customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
#customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"



class App(customtkinter.CTk):

    WIDTH = 1000
    HEIGHT = 520

    def __init__(self):
        super().__init__()

        self.title("CustomTkinter complex example")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        # self.minsize(App.WIDTH, App.HEIGHT)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0,
                                                 bg_color="#002141")
        self.frame_left.grid(row=0, column=0, sticky="nswe")
        self.frame_middle = customtkinter.CTkFrame(master=self)
        self.frame_middle.grid(row=0, column=1, sticky="nswe", padx=(20,5), pady=20)

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=2, sticky="wsn", padx=(5,20), pady=20)

        # ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(7, weight=1)  # empty row as spacing
        #self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.title = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Memory Maker",
                                              text_font=("Roboto Medium", -16),
                                              text_color="#fcfcfc",
                                              bg_color="#0c0c0c",
                                              height=120)  # font name and size in px
        self.title.grid(row=0, column=0, sticky="we")

        self.label_import = customtkinter.CTkLabel(master=self.frame_left,
                                                text="Import",
                                                text_color="#0c0c0c",
                                                bg_color="#fcfcfc",
                                                width=150,
                                                height=70,
                                                text_font=("Roboto Medium", -14))
        self.label_import.grid(row=2, column=0, sticky="w")

        self.label_set = customtkinter.CTkLabel(master=self.frame_left,
                                                text="Set",
                                                text_color="#fcfcfc",
                                                bg_color="#001223",
                                                width=150,
                                                height=70,
                                                text_font=("Roboto Medium", -14))
        self.label_set.grid(row=3, column=0, sticky="w")
        self.label_set.config(state="disabled")

        self.label_render = customtkinter.CTkLabel(master=self.frame_left,
                                                text="Render",
                                                text_color="#fcfcfc",
                                                bg_color="#001223",
                                                width=150,
                                                height=70,
                                                text_font=("Roboto Medium", -14))
        self.label_render.grid(row=4, column=0, sticky="w")
        self.label_render.config(state="disabled")

        self.label_save = customtkinter.CTkLabel(master=self.frame_left,
                                                text="Save Scenes",
                                                text_color="#fcfcfc",
                                                bg_color="#001223",                                                
                                                width=150,
                                                height=70,
                                                text_font=("Roboto Medium", -14))
        self.label_save.grid(row=5, column=0, sticky="w")
        self.label_save.config(state="disabled")

        self.label_finish = customtkinter.CTkLabel(master=self.frame_left,
                                                text="Finish",
                                                text_color="#fcfcfc",
                                                bg_color="#001223",
                                                width=150,
                                                height=70,
                                                text_font=("Roboto Medium", -14))
        self.label_finish.grid(row=6, column=0, sticky="w")
        self.label_finish.config(state="disabled")

       

        # ============ frame_middle ============

        # configure grid layout (3x7)
        self.frame_middle.rowconfigure((0, 1, 2,3), weight=0)
        self.frame_middle.columnconfigure(0, weight=0)
        self.frame_middle.columnconfigure((2,3,4,5), weight=2)
        self.frame_middle.columnconfigure(2, weight=0)

        self.frame_info = customtkinter.CTkFrame(master=self.frame_middle, height=180)
        self.frame_info.grid(row=0, column=0, columnspan=6, rowspan=4, pady=10, padx=10, sticky="nsew")
        self.frame_info.grid_propagate(False)


        # configure grid layout (1x1)
        self.frame_info.rowconfigure(0, weight=1)
        self.frame_info.columnconfigure(0, weight=1)

        # ============ frame_middle ============
        self.color_correction_var = tkinter.IntVar(value=1)
        self.upscaling_var = tkinter.IntVar(value=1)
        self.scene_detection_var = tkinter.IntVar(value=1)

        self.middle_widgets = []

        #Color Correction- Yes / No
        self.label_color_correction = customtkinter.CTkLabel(master=self.frame_middle,
                                                text="Color Correction:",
                                                text_color="#fcfcfc",
                                                width=10,
                                                )
        self.label_color_correction.grid(row=4, column=0, pady=10, padx=5, sticky="w")
        self.middle_widgets.append(self.label_color_correction)

        self.radio_button_1_cc = customtkinter.CTkRadioButton(master=self.frame_middle,
                                                           text="No",
                                                           variable=self.color_correction_var,
                                                           value=0)
        self.radio_button_1_cc.grid(row=4, column=1, pady=10, padx=20, sticky="w")
        self.middle_widgets.append(self.radio_button_1_cc)

        self.radio_button_2_cc = customtkinter.CTkRadioButton(master=self.frame_middle,
                                                           text="Yes",
                                                           variable=self.color_correction_var,
                                                           value=1)
        self.radio_button_2_cc.grid(row=4, column=2, pady=10, padx=20, sticky="w")
        self.middle_widgets.append(self.radio_button_2_cc)

        #Upscaling - Yes / No
        self.label_upscaling = customtkinter.CTkLabel(master=self.frame_middle,
                                                text="Upscaling:",
                                                text_color="#fcfcfc",
                                                width=10,)
        self.label_upscaling.grid(row=5, column=0, pady=10, padx=5, sticky="w")
        self.middle_widgets.append(self.label_upscaling)


        self.radio_button_1_ups = customtkinter.CTkRadioButton(master=self.frame_middle,
                                                           text="No",
                                                           variable=self.upscaling_var,
                                                           value=0)
        self.radio_button_1_ups.grid(row=5, column=1, pady=10, padx=20, sticky="w")
        self.middle_widgets.append(self.radio_button_1_ups)


        self.radio_button_2_ups = customtkinter.CTkRadioButton(master=self.frame_middle,
                                                           text="Yes",
                                                           variable=self.upscaling_var,
                                                           value=1)
        self.radio_button_2_ups.grid(row=5, column=2, pady=10, padx=20, sticky="w")
        self.middle_widgets.append(self.radio_button_2_ups)


        #Scene Detection - Yes / No
        self.label_scene_detection = customtkinter.CTkLabel(master=self.frame_middle,
                                                text="Scene Detection:",
                                                text_color="#fcfcfc",
                                                width=10,
                                                )
        self.label_scene_detection.grid(row=6, column=0, pady=10, padx=5, sticky="w")
        self.middle_widgets.append(self.label_scene_detection)


        self.radio_button_1_sd = customtkinter.CTkRadioButton(master=self.frame_middle,
                                                           text="No",
                                                           variable=self.scene_detection_var,
                                                           value=0)
        self.radio_button_1_sd.grid(row=6, column=1, pady=10, padx=20, sticky="w")
        self.middle_widgets.append(self.radio_button_1_sd)


        self.radio_button_2_sd = customtkinter.CTkRadioButton(master=self.frame_middle,
                                                           text="Yes",
                                                           variable=self.scene_detection_var,
                                                           value=1)
        self.radio_button_2_sd.grid(row=6, column=2, pady=10, padx=20, sticky="w")
        self.middle_widgets.append(self.radio_button_2_sd)


        #Resolution
        self.label_res_entry = customtkinter.CTkLabel(master=self.frame_middle,
                                                text="Upscaling Resolution Width:",
                                                text_color="#fcfcfc",
                                                width=10,
                                                )
        self.label_res_entry.grid(row=7, column=0, pady=10, padx=5, sticky="w")
        self.middle_widgets.append(self.label_res_entry)


        self.res_entry = customtkinter.CTkEntry(master=self.frame_middle,
                                            width=120,
                                            placeholder_text="1080")
        self.res_entry.grid(row=7, column=1, columnspan=3, pady=20, padx=20, sticky="we")
        self.middle_widgets.append(self.res_entry)


        #RENDER BUTTON
        self.btn_render = customtkinter.CTkButton(master=self.frame_middle,
                                                text="Start Render",
                                                command=self.start_render)
        self.btn_render.grid(row=8, column=0, columnspan=6, pady=10, padx=10, sticky="nwe",)
        self.btn_render.configure(state=tkinter.DISABLED)
        self.middle_widgets.append(self.btn_render)

        # ============ frame_right ============
      
        self.btn_open_file = customtkinter.CTkButton(master=self.frame_right,
                                                text="Open File",
                                                command=self.open_file)
        self.btn_open_file.grid(row=0, column=6, columnspan=2, pady=10, padx=10, sticky="nwe",)
        
        #Video Length
        self.label_text_video_length = customtkinter.CTkLabel(master=self.frame_right,
                                                text="Video Length:",
                                                text_color="#fcfcfc",
                                                width=20)
        self.label_text_video_length.grid(row=1, column=6, pady=10, padx=5, sticky="w")
        #self.label_video_length.place(relx=0, rely=0.15, anchor="w")

        self.label_val_video_length = customtkinter.CTkLabel(master=self.frame_right,
                                                text="25 sec",
                                                text_color="#fcfcfc",
                                                width=20)
        self.label_val_video_length.grid(row=1, column=7, pady=10, padx=5, sticky="w")

        #Resolution
        self.label_text_video_res = customtkinter.CTkLabel(master=self.frame_right,
                                                text="Resolution:",
                                                text_color="#fcfcfc",
                                                width=20)
        self.label_text_video_res.grid(row=2, column=6, pady=10, padx=5, sticky="w")

        self.label_val_video_res = customtkinter.CTkLabel(master=self.frame_right,
                                                text="1920x1080",
                                                text_color="#fcfcfc",
                                                width=20)
        self.label_val_video_res.grid(row=2, column=7, pady=10, padx=5, sticky="w")

        #FPS
        self.label_text_video_fps = customtkinter.CTkLabel(master=self.frame_right,
                                                text="FPS:",
                                                text_color="#fcfcfc",
                                                width=20)
        self.label_text_video_fps.grid(row=3, column=6, pady=10, padx=5, sticky="w")

        self.label_val_video_fps = customtkinter.CTkLabel(master=self.frame_right,
                                                text="30",
                                                text_color="#fcfcfc",
                                                width=20)
        self.label_val_video_fps.grid(row=3, column=7, pady=10, padx=5, sticky="w")

    def button_event(self):
        print("Button pressed")

    def on_closing(self, event=0):
        self.destroy()

    def start(self):
        self.mainloop()

    def open_file(self):
        global image_preview
        app.filename = filedialog.askopenfilenames(initialdir=os.getcwd()+"/..",
                                                            title="Select a File", filetypes=[
                                                    ("all video format", ".mp4"),
                                                    ("all video format", ".flv"),
                                                    ("all video format", ".avi"),
                                                            ])
        
        print(extract_frames(app.filename[0], app.filename[0]))
        app.image_preview = ImageTk.PhotoImage(random_image.resize((256,144), Image.Resampling.BICUBIC))
        app.label_image_preview = customtkinter.CTkLabel(master=app.frame_info,
                                                    image=app.image_preview,
                                                    fg_color="#0c0c0c",  # <- custom tuple-color
                                                    justify=tkinter.LEFT)
        app.label_image_preview.grid(column=0, row=0, sticky="nwe", padx=5, pady=5)


        app.label_image_filepath = customtkinter.CTkLabel(master=app.frame_info,
                                                    text="Video Path: " + app.filename[0],
                                                    width=20)
        app.label_image_filepath.grid(column=0, row=1, sticky="w", padx=3, pady=3)
        app.middle_widgets.append(app.label_image_filepath)
        app.btn_render.configure(state=tkinter.NORMAL)
        self.label_set.config(state="normal")
        self.label_set.config(bg_color="#fcfcfc")
        self.label_set.config(text_color="#0c0c0c")
        app.label_import.config(bg_color="#001223")
        app.label_import.config(state="disabled")

    def start_render(self):
        app.label_set.config(text_color="#fcfcfc")
        app.label_set.config(bg_color="#001223")
        self.label_set.config(state="disabled")

        app.label_render.config(text_color="#0c0c0c")
        app.label_render.config(bg_color="#fcfcfc")
        app.label_render.config(state="normal")
        #app.frame_middle.grid_remove()
        for widget in app.middle_widgets:
            widget.destroy()
        app.frame_right.grid_remove()
        self.progress_bar = customtkinter.CTkProgressBar(master=app.frame_info)
        self.progress_bar.grid(row=1, column=0,padx=10, pady=10, sticky="ew")
        self.progress_bar.configure(bg_color="#021A31",
                        progress_color="#008E2B",)
        #Progress in % percentage / 100
        self.progress_bar.set(0.64)
        predict_images("./Images")
        
        

        '''self.frame_right_r = customtkinter.CTkFrame(master=self)
        self.frame_right_r.grid(row=0, column=1, columnspan=4, sticky="wsn", padx=(5,20), pady=20)
        self.frame_right_r2 = customtkinter.CTkFrame(master=self)
        self.frame_right_r2.grid(row=0, column=2, columnspan=4, sticky="wsn", padx=(5,20), pady=20)

        self.frame_info_r = customtkinter.CTkFrame(master=self.frame_right_r,)
        self.frame_info_r.grid(row=0, column=0, columnspan=4, rowspan=4, 
        pady=10, padx=10, sticky="wsn")
        self.frame_info_l = customtkinter.CTkFrame(master=self.frame_right_r,)
        self.frame_info_l.grid(row=0, column=1, #columnspan=6, rowspan=4, 
        pady=10, padx=10, sticky="wsn")
    '''
        



    



def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using decord's VideoReader
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """
    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    # load the VideoReader
    vr = VideoReader(video_path, ctx=cpu(0))  # can set to cpu or gpu .. ctx=gpu(0)
                     
    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = len(vr)

    frames_index_list = list(range(start, end, every))
    saved_count = 0

    global frames 
    frames = []
    
    if every > 25 and len(frames_index_list) < 1000:  # this is faster for every > 25 frames and can fit in memory
        frames = vr.get_batch(frames_index_list).asnumpy()
        
    else:  # this is faster for every <25 and consumes small memory
        for index in range(start, end):  # lets loop through the frames until the end
            frame = vr[index]  # read an image from the capture
            
            if index % every == 0:  # if this is a frame we want to write out based on the 'every' argument
                frames.append(frame.asnumpy())
    
    
    print(len(frames))
    print(frames[0])
    print(type(frames[0]))
    
    from model import predict_frames
    predict_frames(frames)
    
    import timeit
    #frames = vr.get_batch(frames_index_list).asnumpy()
    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    print(type(vr[randint(start,end)]))
    print(type(vr.next().asnumpy()))
    global random_image
    random_image = Image.fromarray(vr.next().asnumpy(), 'RGB')
    wid, hgt = random_image.size
    print(wid, hgt)
    print("FRames: ", end)


    my_formatter = "{0:.2f}"

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    app.label_val_video_length['text'] = my_formatter.format(end/fps) + "s"
    app.label_val_video_res['text'] = str(wid) + "x" + str(hgt)
    app.label_val_video_fps['text'] = my_formatter.format(fps)
    print("The time difference is :", timeit.default_timer() - starttime)
    print("The end time is :",timeit.default_timer())
    return saved_count  # and return the count of the images we saved


def render():
    pass
    #for frame in frames:
        #first step color
        #upscale
        #ai prediction
            #scene listener -> if scene changes and ai continues to make different prediction it is a new scene
            #get scene name -> e.g 70% buildings 30% street -> City or > 50% sea -> Sea, > 90% street -> highway

if __name__ == "__main__":
    app = App()
    app.start()