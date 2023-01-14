import os
import sys
from tkinter import *
from tkinter import filedialog, ttk, messagebox
import configparser, webbrowser
import math

class TextEdit:
    def __init__(self, root):
        root.title(self.__class__.__name__)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)


        frame1 = Frame(root)
        frame2 = Frame(root)

        # button_1のハンドラ関数
        def change_no2():
            frame2.tkraise() # frame_2を前面に出す

        # button_2のハンドラ関数
        def change_no1():
            frame1.tkraise() # frame_1を前面に出す
        
        # ハンドラ関数
        def click_get():
            messagebox.showinfo('メッセージ', var.get())

        btn1 = ttk.Button(frame1, text='押してください！', command=self.button1Clicked)
        btn2 = ttk.Button(frame1, text='押してください！2', command=self.button1Clicked)
        btn3 = ttk.Button(frame1, text='次の画面', command=change_no2)
        # btn1.pack(pady=20)



        txt = Entry(frame1, width=20)
        # txt.pack(pady=20)
        # ハンドラ関数
        def click(event):
            messagebox.showinfo('メッセージ', txt.get())

        # Labelウィジェットの生成
        label = Label(frame1, text='ここをクリック', foreground='red')

        # Labelウィジェットの配置
        # label.pack()
        # ハンドラ関数を設定
        label.bind("<Button-1>", click)

        action = ['選択肢1', '選択肢2', '選択肢3', '選択肢4']
        # 選択状態を保持する変数（初期値を'選択肢1'にしている）
        var = StringVar(value='選択肢1')
        radio1 = Radiobutton(root, text=action[0], variable=var, value=action[0])
        radio2 = Radiobutton(root, text=action[1], variable=var, value=action[1])
        radio3 = Radiobutton(root, text=action[2], variable=var, value=action[2])
        radio4 = Radiobutton(root, text=action[3], variable=var, value=action[3])

        # Buttonウィジェットの生成と配置
        btn4 = Button(frame1, text='表示', command=click_get)

        # grid関数でウィジェットを配置
        frame1.grid(row=0, column=0, sticky=NSEW)
        btn1.grid(column=0, row=0, sticky=E)
        txt.grid(column=1, row=0)
        label.grid(column=2, row=0, sticky=W)
        btn2.grid(column=0, row=1, sticky=E)
        btn3.grid(column=1, row=1, sticky=E)
        btn4.grid(column=2, row=1, sticky=E)
        radio1.grid(column=0, row=2, sticky=E)
        radio2.grid(column=1, row=2, sticky=E)
        radio3.grid(column=2, row=2, sticky=E)


        frame2.grid(row=0, column=0, sticky=NSEW)
        label2 = Label(frame2, text='画面2です。')
        button2 = Button(frame2, text='戻る',  command=change_no1)
        label2.pack(pady=20)
        button2.pack(pady=20)

        frame1.tkraise()

        
        
        
        self.fileTypes = [('テキストファイル', '*.txt'), ('すべてのファイル','*.*')]
        self.directory = os.getenv('HOMEDRIVE') + os.getenv('HOMEPATH') + '\\Documents'

        clientHeight = '50'
        clientWidth = '300'
        cp = configparser.ConfigParser()
        try:
            cp.read(self.__class__.__name__ + '.ini')
            clientHeight = cp['Client']['Height']
            clientWidth = cp['Client']['Width']
        except:
            print(self.__class__.__name__ + ':Use default value(s)', file=sys.stderr)
        
        root.geometry(clientWidth + 'x' + clientHeight)
        root.protocol('WM_DELETE_WINDOW', self.menuFileExit)

        root.option_add('*tearOff', FALSE)
        menu = Menu()
        menuFile = Menu()
        menu.add_cascade(menu=menuFile, label='ファイル(F)', underline=5)
        menuFile.add_command(label='新規(N)', underline=3, command=self.menuFileNew)
        menuFile.add_command(label='開く(O)', underline=3, command=self.menuFileOpen)
        menuFile.add_command(label='保存(S)', underline=3, command=self.menuFileSave)
        menuFile.add_command(label='名前を付けてシフトJISで保存(A)', underline=16, command=self.menuFileSaveAsSjis)
        menuFile.add_command(label='名前を付けてUTF-8で保存(U)', underline=15, command=self.menuFileSaveAsUtf8)
        menuFile.add_separator()
        menuFile.add_command(label='終了(x)', underline=3, command=self.menuFileExit)
        menuHelp = Menu()
        menu.add_cascade(menu=menuHelp, label='ヘルプ(H)', underline=4)
        menuHelp.add_command(label='バージョン情報(v)', underline=8, command=self.menuHelpVersion)
        root['menu'] = menu
    
    def menuFileNew(self):
        self.isSjis = TRUE
        self.textFilename = ''
        self.text.delete('1.0', 'end')

    def menuFileOpen(self):
        filename = filedialog.askopenfilename(filetypes=self.fileTypes, initialdir=self.directory)
        if not filename:
            return
        
        newText = ''
        try:
            f = open(filename, 'r')
            newText = f.read()
            self.isSjis = TRUE
        except:
            f = open(filename, 'r', encoding='UTF-8')
            newText = f.read()
            self.isSjis = FALSE
        finally: 
            f.close()

        if newText == '':
            messagebox.showwarning(self.__class__.__name__, 'ファイルを開けませんでした')
        else:
            self.text.delete('1.0', 'end')
            self.text.insert('1.0', newText)
            self.textFilename = filename

    def menuFileSave(self):
        self.fileSave(self.textFilename, self.isSjis)

    def fileSave(self, saveFilename, saveIsSjis):
        s = self.text.get('1.0', 'end')
        if len(s) == 1:
            messagebox.showwarning(self.__class__.__name__, '保存するテキストがありません')
            return

        if saveIsSjis == TRUE:
            f = open(saveFilename, 'w')
        else:
            f = open(saveFilename, 'w', encoding='UTF-8')
        f.write(s[:-1])
        f.close()
        self.isSjis = saveIsSjis
        self.textFilename = saveFilename

    def menuFileSaveAsSjis(self):
        self.fileSaveAs(TRUE)

    def menuFileSaveAsUtf8(self):
        self.fileSaveAs(FALSE)

    def fileSaveAs(self, saveIsSjis):
        filename = filedialog.asksaveasfilename(
            filetypes=self.fileTypes,
            initialdir=self.directory,
            initialfile=os.path.basename(self.textFilename))
        if not filename:
            return
        self.fileSave(filename, saveIsSjis)
    
    def menuHelpVersion(self):
        s = self.__class__.__name__
        s += ' Version 0.01(2022/12/27)\n'
        s += '©2022 Tomohiro K\n'
        s += 'with Python ' + sys.version
        messagebox.showinfo(self.__class__.__name__, s)
    
    def menuFileExit(self):
        cp = configparser.ConfigParser()
        cp['Client'] = {
            'Height': str(root.winfo_height()),
            'Width': str(root.winfo_width())}
        cp['File'] = {'Directory': self.directory}
        with open(self.__class__.__name__ + '.ini', 'w') as f:
            cp.write(f)
        root.destroy()
    
    def button1Clicked(self):
        messagebox.showinfo(root.title(), 'ありがとうございます！')

root = Tk()
TextEdit(root)
# print(root.children)
root.mainloop()



