# Jeu de Dames

## Description
This project is a digital implementation of the classic board game "Dames" (Checkers). It allows two players to compete against each other, either through a graphical interface or via a command line chat interface. The game includes features such as piece movement, capturing opponent pieces, and a chat functionality for player communication.

## Files Overview

- **Dame.py**: Defines the `Dame` class, which represents a queen piece in the game. It includes methods for initialization, display, possible moves, and capturing logic.

- **Case.py**: Defines the `Case` class, representing a square on the game board. It includes attributes for position (x, y), color, and a reference to any piece occupying the square.

- **Regles.py**: Contains the `Regles` class with static methods for validating moves in the game, including rules for capturing and movement.

- **Mouvement.py**: Defines the `Mouvement` class, which represents a move in the game with attributes for the starting and ending positions.

- **Joueur.py**: Defines the `Joueur` class, representing a player in the game, with an attribute for the color of their pieces.

- **Interface.py**: An abstract base class that outlines methods for displaying the game board, requesting moves, and displaying messages.

- **Jeu.py**: Manages the game logic, including initialization, user input handling, and game state updates.

- **ChatInterface.py**: Creates a graphical user interface for chat functionality, allowing users to send messages and restart the game.

- **Main.py**: The entry point for the application, initializing the chat interface and starting the game in a separate thread.

- **Instance.py**: Defines the `Instance` class, which holds a list of positions of the pieces (both black and white) and includes a method `run_to_the_end` to execute random moves until one player wins.

## Setup and Usage

1. **Requirements**: Ensure you have Python installed on your machine.

2. **Installation**: Clone the repository or download the project files.

3. **Running the Game**:
   - Navigate to the project directory in your terminal.
   - Run the `Main.py` file to start the game and chat interface:
     ```
     python Main.py
     ```

4. **Gameplay**: Follow the on-screen instructions to play the game. Use the chat interface to communicate with your opponent.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.