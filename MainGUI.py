import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import configparser, webbrowser
import math
import torch
import pandas as pd

class TextEdit:
    def __init__(self, root):
        root.title(self.__class__.__name__)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=1)
        root.columnconfigure(3, weight=1)
        root.rowconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=1)
        root.rowconfigure(3, weight=1)
        root.rowconfigure(4, weight=1)
        root.rowconfigure(5, weight=1)
        root.rowconfigure(6, weight=1)
        root.rowconfigure(7, weight=1)
        
        self.count = 0
        
        
        # ハンドラ関数
        def get_entry1():
            messagebox.showinfo('メッセージ', entry1.get())
        
        def get_combo1():
            messagebox.showinfo('メッセージ', combo1.get())
        
        def get_radio1():
            messagebox.showinfo('メッセージ', action[var.get()])
        
        def count_up():
            # global count
            self.count = self.count + 1
            label1['text'] = str(self.count)
            
        weather = ('晴れ','曇り','雨')
        
        # Labelウィジェットの生成
        label1 = tk.Label(root, text='Pytorch環境を確認', foreground='red')
        label2 = tk.Label(root, text='データ読み込み', foreground='red')
        self.label3 = tk.Label(root, text='trainデータパス', foreground='red')
        self.label4 = tk.Label(root, text='testデータパス', foreground='red')
        
        
        # Entryウィジェットの生成
        entry1 = tk.Entry(width=20)
        
        # Buttonウィジェットの生成
        button1 = tk.Button(root, text='Pytorch環境', command=self.check_pytorch)
        button2 = tk.Button(root, text='trainデータ読み込み', command=self.trainFileOpen)
        button3 = tk.Button(root, text='testデータ読み込み', command=self.testFileOpen)
        # button1 = tk.Button(root, text='ボタン1', command=count_up)
        # button2 = tk.Button(root, text='表示', command=get_entry1)
        button98 = tk.Button(root, text='コンボ', command=get_combo1)
        button99 = tk.Button(root, text='コンボ', command=get_radio1)
        
        # Comboウィジェットの生成
        combo1 = ttk.Combobox(root, state='readonly', values=weather)
        
        

        action = ['選択肢1', '選択肢2', '選択肢3', '選択肢4']
        var = tk.IntVar(value=0)
        radio1 = tk.Radiobutton(root, text=action[0], variable=var, value=0)
        radio2 = tk.Radiobutton(root, text=action[1], variable=var, value=1)
        radio3 = tk.Radiobutton(root, text=action[2], variable=var, value=2)
        radio4 = tk.Radiobutton(root, text=action[3], variable=var, value=3)

        # Buttonウィジェットの生成と配置
        # btn4 = tk.Button(root, text='表示', command=click_get)

        # grid関数でウィジェットを配置
        label1.grid(row=0, column=0, sticky=tk.E)
        button1.grid(row=0, column=1, sticky=tk.E)
        label2.grid(row=1, column=0, sticky=tk.N)
        button2.grid(row=1, column=1, sticky=tk.S)
        self.label3.grid(row=1, column=2, sticky=tk.S)
        button3.grid(row=2, column=1, sticky=tk.S)
        self.label4.grid(row=2, column=2, sticky=tk.S)
        combo1.grid(row=5, column=1, sticky=tk.W)
        button98.grid(row=5, column=2, sticky=tk.NSEW)
        radio1.grid(row=6, column=0)
        radio2.grid(row=6, column=1)
        radio3.grid(row=6, column=2)
        radio4.grid(row=6, column=3)
        button99.grid(row=7, column=3, sticky=tk.NSEW)
        # entry1.grid(row=1, column=1, sticky=tk.W)
        # button1.grid(row=1, column=2, sticky=tk.E)
        # button2.grid(row=1, column=3, sticky=tk.NSEW)
        

        
        
        self.fileTypes = [('csvファイル', '*.csv'), ('すべてのファイル','*.*')]
        self.directory = os.getenv('HOMEDRIVE') + os.getenv('HOMEPATH') + '\\Documents'

        clientHeight = '50'
        clientWidth = '300'
        cp = configparser.ConfigParser()
        try:
            cp.read(self.__class__.__name__ + '.ini')
            clientHeight = cp['Client']['Height']
            clientWidth = cp['Client']['Width']
            self.directory =  cp['File']['Directory']
        except:
            print(self.__class__.__name__ + ':Use default value(s)', file=sys.stderr)
        
        root.geometry(clientWidth + 'x' + clientHeight)
        root.protocol('WM_DELETE_WINDOW', self.menuFileExit)

        # root.option_add('*tearOff', FALSE)
        menu = tk.Menu()
        menuFile = tk.Menu()
        menu.add_cascade(menu=menuFile, label='ファイル(F)', underline=5)
        # menuFile.add_separator()
        menuFile.add_command(label='終了(x)', underline=3, command=self.menuFileExit)
        menuHelp = tk.Menu()
        menu.add_cascade(menu=menuHelp, label='ヘルプ(H)', underline=4)
        menuHelp.add_command(label='バージョン情報(v)', underline=8, command=self.menuHelpVersion)
        root['menu'] = menu
    

    def menuHelpVersion(self):
        s = self.__class__.__name__
        s += ' Version 0.01(2023/3/12)\n'
        s += '©2023 Tomohiro K\n'
        s += 'with Python ' + sys.version
        messagebox.showinfo(self.__class__.__name__, s)
        
    def menuFileExit(self):
        cp = configparser.ConfigParser()
        cp['Client'] = {
            'Height': str(root.winfo_height()),
            'Width': str(root.winfo_width())}
        cp['File'] = {
            'Directory': self.directory}
        with open(self.__class__.__name__ + '.ini', 'w') as f:
            cp.write(f)
        root.destroy()
    
    def check_pytorch(self):
        cpt = 'GPU available : ' + str(torch.cuda.is_available()) + '\n'
        cpt += 'Pytorch version : ' +  torch.__version__ + '\n'
        cpt += 'GPU count : ' + str(torch.cuda.device_count()) + '\n'
        cpt += 'GPU index : ' + str(torch.cuda.current_device()) + '\n'
        cpt += 'GPU device name : ' + torch.cuda.get_device_name() + '\n'
        cpt += 'GPU device capability : ' + str(torch.cuda.get_device_capability())
        messagebox.showinfo('Pytorch環境', cpt)

    def trainFileOpen(self):
        try:
            filepath = filedialog.askopenfilename(filetypes=self.fileTypes, initialdir=self.directory)
        except:
            filepath = ''
        
        try:
            train_data = pd.read_csv(filepath)
            self.trainFile = train_data # csvファイルの中身
            self.trainFilePath = filepath
            self.directory = filepath
        except:
            messagebox.showwarning(self.__class__.__name__, 'ファイルを開けませんでした')
        
        self.label3.configure(text=filepath)
    
    def testFileOpen(self):
        try:
            filepath = filedialog.askopenfilename(filetypes=self.fileTypes, initialdir=self.directory)
        except:
            filepath = ''
        
        try:
            train_data = pd.read_csv(filepath)
            self.testnFile = train_data # csvファイルの中身
            self.testFilePath = filepath
            self.directory = filepath
        except:
            messagebox.showwarning(self.__class__.__name__, 'ファイルを開けませんでした')
        
        self.label4.configure(text=filepath)
        


root = tk.Tk()
TextEdit(root)
# print(root.children)
root.mainloop()



