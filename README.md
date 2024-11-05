
# Smart Pokedex 
## 🚀 Requirements

Before you begin, ensure you have the following tools installed:

- [Docker](https://www.docker.com/get-started) ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
- [Visual Studio Code](https://code.visualstudio.com/) ![VSCode](https://img.shields.io/badge/VSCode-007ACC?logo=visual-studio-code&logoColor=white)
- [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) ![Remote - Containers](https://img.shields.io/badge/Remote%20Containers-1E1E1E?logo=visual-studio-code&logoColor=white)
- [Poetry](https://python-poetry.org/docs/#installation) ![Poetry](https://img.shields.io/badge/Poetry-8CC84B?logo=python&logoColor=white) (if not using Dev Container)

## 🛠 Setting Up Environment Using Dev Container

1. **Clone the repository**

   First, clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. **Start the Dev Container**

   Open the project folder in Visual Studio Code. When you open the folder, VSCode should automatically suggest reopening it in a Dev Container. You can also do this manually:

   - Open the command palette (`Ctrl + Shift + P`).
   - Select `Remote-Containers: Reopen Folder in Container`.

   VSCode will download the necessary Docker image and create a container. Once completed, you will have access to a fully configured environment.

3. **Install dependencies**

   If you have a `poetry.lock` file in your project, all dependencies will be automatically installed when the container is opened. If you want to install new dependencies, use:

   ```bash
   poetry add <package_name>
   ```

## 🛠 Setting Up Environment Using Poetry (Without Dev Container)

If you prefer not to use a Dev Container, you can set up your environment locally using Poetry:

1. **Install Poetry**

   If you haven't installed Poetry yet, you can do so with the following command:

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   Make sure to add `~/.local/bin` to your `$PATH`.

2. **Clone the repository**

   Copy the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

3. **Install dependencies**

   Install all project dependencies:

   ```bash
   poetry install
   ```

4. **Running the project**

   To run the project, you can use:

   ```bash
   poetry run python <filename.py>
   ```

## 🐞 Reporting Issues

If you encounter any problems, please open an issue in the repository or contact the project author.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
