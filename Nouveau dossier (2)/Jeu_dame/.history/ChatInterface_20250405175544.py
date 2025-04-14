import tkinter as tk
from tkinter.scrolledtext import ScrolledText

class ChatInterface:
    def __init__(self):
        """
        Initialise l'interface de chat.
        """
        self.root = tk.Tk()
        self.root.title("Chat - Jeu de Dames")

        # Zone de texte pour afficher les messages
        self.chat_display = ScrolledText(self.root, state='disabled', wrap='word', height=20, width=50)
        self.chat_display.grid(row=0, column=0, padx=10, pady=10)

        # Zone de saisie pour envoyer des messages (facultatif)
        self.entry = tk.Entry(self.root, width=40)
        self.entry.grid(row=1, column=0, padx=10, pady=5)
        self.entry.bind("<Return>", self.send_message)

    def add_message(self, message):
        """
        Ajoute un message à la zone de chat.
        :param message: Le message à afficher.
        """
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, message + "\n")
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)  # Faire défiler automatiquement vers le bas

    def send_message(self, event=None):
        """
        Gère l'envoi d'un message (facultatif).
        """
        message = self.entry.get()
        if message:
            self.add_message(f"Vous: {message}")
            self.entry.delete(0, tk.END)

    def run(self):
        """
        Lance la boucle principale de l'interface.
        """
        self.root.mainloop()