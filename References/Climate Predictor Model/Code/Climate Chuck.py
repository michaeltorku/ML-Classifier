import tkinter
from tkinter import *

def Main_Window():
    global Main

    Main = Tk()

    Main.title("Team Two Product")
    Main.geometry("650x400")
    Main.resizable(0,0)
    Main.configure(background = "SkyBlue1")

    # Widgets

    MainTitle_label = tkinter.Label(Main, text = " Climate Chuck ", bg = "royalblue", height = 2 )
    MainTitle_label.config(font = ("System", 32))
    MainTitle_label.pack(fill = X)

    Space_label = tkinter.Label(Main, text = " ", fg = "SkyBlue1", bg = "SkyBlue1", height = 2 , width = 36, padx = 1, pady = 1)
    Space_label.pack()

    Forecast_btn = tkinter.Button(Main, text = "Forecast", command = WeatherPredictionWindow, fg = "black", bg = "white", height = 2 , width = 24, padx = 1, pady = 1)
    Forecast_btn.config(font = ("Arial", 16))
    Forecast_btn.pack()

    Space_label = tkinter.Label(Main, text = " ", fg = "SkyBlue1", bg = "SkyBlue1", height = 2 , width = 36, padx = 1, pady = 1)
    Space_label.pack()
    
    Information_btn = tkinter.Button(Main, text = "Forecast Information", command = InformationWindow, fg = "black", bg = "white", height = 2 , width = 24, padx = 1, pady = 1)
    Information_btn.config(font = ("Arial", 16))
    Information_btn.pack()
    
    #This starts the window

    Main.mainloop()


def WeatherPredictionWindow():
    
    PredWindow = Tk()
    Main.destroy()
    
    PredWindow.title("Team Two Product")
    PredWindow.geometry("650x400")
    PredWindow.resizable(0,0)
    PredWindow.configure(background = "SkyBlue1")

    # Widgets
    BorrowTitle_label = tkinter.Label(PredWindow, text = " Weather Forecast ", fg = "black", bg = "white", padx = 5, pady = 5)
    BorrowTitle_label.config(font = ("System", 32))
    BorrowTitle_label.grid(row = 0, columnspan = 8)    

    BorrowTitle_label = tkinter.Label(PredWindow, text = "  ", fg = "black", bg = "white", padx = 5, pady = 5)
    BorrowTitle_label.config(font = ("System", 32))
    BorrowTitle_label.grid(row = 0, columnspan = 8)

    
    
    Location_label = tkinter.Label(PredWindow, text = "   Location   ", fg = "SkyBlue1", bg = "SkyBlue1")
    Location_label.grid(row = 2, column = 0)

    Location_entry = tkinter.Entry(PredWindow)
    Location_entry.grid(row = 2, column = 1)

    
    

def InformationWindow():
    print("Infromation")

Main_Window()
